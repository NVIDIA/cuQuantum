# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Expectation value with gradients for a 4-qubit circuit.

Circuit:
  q0 --- Ry [grad] -------------------- CNOT(0,1) ---
  q1 --- H --- CNOT(1,2) --- Rx ---------------------
  q2 ---------------------- CNOT(2,3) -- Rz [grad] ---
  q3 --- H -----------------------------------------

Hamiltonian: H = 2.0*XYZZ + 3.0*IZZI + 5.0*ZIYY
Gradients are computed for Ry on q0 and Rz on q2.
"""

import cupy as cp
import numpy as np

import cuquantum
from cuquantum.bindings import cutensornet as cutn


print("cuTensorNet-vers:", cutn.get_version())
dev = cp.cuda.Device()
props = cp.cuda.runtime.getDeviceProperties(dev.id)
print("===== device info ======")
print("GPU-local-id:", dev.id)
print("GPU-name:", props["name"].decode())
print("========================")

#######################################
# Expectation with gradient computation
#######################################

num_qubits = 4
qubits_dims = (2,) * num_qubits
theta = np.pi / 4.0
print(f"Quantum circuit: {num_qubits} qubits (expectation gradient)")

#############
# cuTensorNet
#############

handle = cutn.create()
stream = cp.cuda.Stream()
data_type = cuquantum.cudaDataType.CUDA_C_64F

# Gate tensors matching C++ (column-major / Fortran order)
inv_sqrt2 = 1.0 / np.sqrt(2.0)
gate_h = np.array(
    [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype="complex128", order="F"
)
gate_h = cp.asarray(gate_h)

# CNOT as 4x4 in basis (|00>,|01>,|10>,|11>); order="F" so layout matches C++ backend.
gate_cx = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype="complex128",
    order="F",
)
gate_cx = cp.asarray(gate_cx.reshape(2, 2, 2, 2, order="F"))

# RX(theta)
c, s = np.cos(theta / 2.0), -np.sin(theta / 2.0)
gate_rx = np.array([[c, 1j * s], [1j * s, c]], dtype="complex128", order="F")
gate_rx = cp.asarray(gate_rx)

# RY(theta)
c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
gate_ry = np.array([[c, s], [-s, c]], dtype="complex128", order="F")
gate_ry = cp.asarray(gate_ry)

# RZ(theta)
gate_rz = np.array(
    [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
    dtype="complex128",
    order="F",
)
gate_rz = cp.asarray(gate_rz)

# Pauli X, Y, Z, I
gate_x = np.array([[0, 1], [1, 0]], dtype="complex128", order="F")
gate_x = cp.asarray(gate_x)
gate_y = np.array([[0, -1j], [1j, 0]], dtype="complex128", order="F")
gate_y = cp.asarray(gate_y)
gate_z = np.array([[1, 0], [0, -1]], dtype="complex128", order="F")
gate_z = cp.asarray(gate_z)
gate_i = np.eye(2, dtype="complex128", order="F")
gate_i = cp.asarray(gate_i)

# Gradient output buffers (allocated before applying gates with gradient)
d_grad_ry = cp.zeros(4, dtype="complex128")
d_grad_rz = cp.zeros(4, dtype="complex128")
print("Allocated gradient buffers on GPU")

# Scratch workspace
free_mem = dev.mem_info[0]
scratch_size = (free_mem - (free_mem % 4096)) // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the initial quantum state
quantum_state = cutn.create_state(
    handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type
)
print("Created the initial quantum state")

# Apply gates
cutn.state_apply_tensor_operator(
    handle, quantum_state, 1, (1,), int(gate_h.data.ptr), 0, 1, 0, 1
)
cutn.state_apply_tensor_operator(
    handle, quantum_state, 1, (3,), int(gate_h.data.ptr), 0, 1, 0, 1
)
cutn.state_apply_tensor_operator(
    handle, quantum_state, 2, (1, 2), int(gate_cx.data.ptr), 0, 1, 0, 1
)
cutn.state_apply_tensor_operator(
    handle, quantum_state, 2, (2, 3), int(gate_cx.data.ptr), 0, 1, 0, 1
)
cutn.state_apply_tensor_operator_with_gradient(
    handle,
    quantum_state,
    1,
    (0,),
    int(gate_ry.data.ptr),
    0,
    0,
    0,
    1,
    int(d_grad_ry.data.ptr),
    0,
)
cutn.state_apply_tensor_operator(
    handle, quantum_state, 1, (1,), int(gate_rx.data.ptr), 0, 0, 0, 1
)
cutn.state_apply_tensor_operator_with_gradient(
    handle,
    quantum_state,
    1,
    (2,),
    int(gate_rz.data.ptr),
    0,
    0,
    0,
    1,
    int(d_grad_rz.data.ptr),
    0,
)
cutn.state_apply_tensor_operator(
    handle, quantum_state, 2, (0, 1), int(gate_cx.data.ptr), 0, 1, 0, 1
)
print("Applied quantum gates (Ry on q0 and Rz on q2 registered for gradient)")

# Hamiltonian: 2.0*XYZZ + 3.0*IZZI + 5.0*ZIYY
hamiltonian = cutn.create_network_operator(handle, num_qubits, qubits_dims, data_type)
num_modes_4 = (1, 1, 1, 1)
state_modes_0123 = ((0,), (1,), (2,), (3,))

cutn.network_operator_append_product(
    handle,
    hamiltonian,
    np.complex128(2.0),
    4,
    num_modes_4,
    state_modes_0123,
    0,
    (gate_x.data.ptr, gate_y.data.ptr, gate_z.data.ptr, gate_z.data.ptr),
)
cutn.network_operator_append_product(
    handle,
    hamiltonian,
    np.complex128(3.0),
    4,
    num_modes_4,
    state_modes_0123,
    0,
    (gate_i.data.ptr, gate_z.data.ptr, gate_z.data.ptr, gate_i.data.ptr),
)
cutn.network_operator_append_product(
    handle,
    hamiltonian,
    np.complex128(5.0),
    4,
    num_modes_4,
    state_modes_0123,
    0,
    (gate_z.data.ptr, gate_i.data.ptr, gate_y.data.ptr, gate_y.data.ptr),
)
print("Constructed a tensor network operator: 2.0*XYZZ + 3.0*IZZI + 5.0*ZIYY")

# Create expectation, configure (numHyperSamples=8), prepare, set workspace
expectation = cutn.create_expectation(handle, quantum_state, hamiltonian)
print("Created the specified quantum circuit expectation value")

# Configure num_hyper_samples
num_hyper_samples = np.array(8, dtype=np.int32)
cutn.expectation_configure(
    handle,
    expectation,
    cutn.ExpectationAttribute.CONFIG_NUM_HYPER_SAMPLES,
    num_hyper_samples.ctypes.data,
    num_hyper_samples.nbytes,
)

work_desc = cutn.create_workspace_descriptor(handle)
cutn.expectation_prepare(handle, expectation, scratch_size, work_desc, stream.ptr)
print("Created the workspace descriptor")
print("Prepared the specified quantum circuit expectation value (gradient backward)")

# Query SCRATCH workspace size
workspace_scratch_size = cutn.workspace_get_memory_size(
    handle,
    work_desc,
    cutn.WorksizePref.RECOMMENDED,
    cutn.Memspace.DEVICE,
    cutn.WorkspaceKind.SCRATCH,
)
print(f"Required scratch GPU workspace size (bytes) = {workspace_scratch_size}")

if workspace_scratch_size <= scratch_size:
    cutn.workspace_set_memory(
        handle,
        work_desc,
        cutn.Memspace.DEVICE,
        cutn.WorkspaceKind.SCRATCH,
        scratch_space.ptr,
        workspace_scratch_size,
    )
else:
    print("ERROR: Insufficient scratch workspace size on Device!")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_expectation(expectation)
    cutn.destroy_network_operator(hamiltonian)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch_space
    exit(1)
print("Set the workspace buffer")

# Compute expectation and gradients (backward)
expectation_value = np.empty(1, dtype=np.complex128)
expectation_adjoint = np.array([1.0 + 0.0j], dtype=np.complex128)
cutn.expectation_compute_with_gradients_backward(
    handle,
    expectation,
    0,
    expectation_adjoint.ctypes.data,
    0,
    work_desc,
    expectation_value.ctypes.data,
    0,
    stream.ptr,
)

stream.synchronize()
print("Computed the specified quantum circuit expectation value and gradients")

ev = expectation_value.item()
print(f"Expectation value = ({ev.real}, {ev.imag})")

# Copy gradients from device to host
h_grad_ry = cp.asnumpy(d_grad_ry)
h_grad_rz = cp.asnumpy(d_grad_rz)
print("Gradient d<H>/d(Ry on q0):")
print(f"  [0,0]: ({h_grad_ry[0].real}, {h_grad_ry[0].imag})")
print(f"  [0,1]: ({h_grad_ry[1].real}, {h_grad_ry[1].imag})")
print(f"  [1,0]: ({h_grad_ry[2].real}, {h_grad_ry[2].imag})")
print(f"  [1,1]: ({h_grad_ry[3].real}, {h_grad_ry[3].imag})")
print("Gradient d<H>/d(Rz on q2):")
print(f"  [0,0]: ({h_grad_rz[0].real}, {h_grad_rz[0].imag})")
print(f"  [0,1]: ({h_grad_rz[1].real}, {h_grad_rz[1].imag})")
print(f"  [1,0]: ({h_grad_rz[2].real}, {h_grad_rz[2].imag})")
print(f"  [1,1]: ({h_grad_rz[3].real}, {h_grad_rz[3].imag})")

# Cleanup
cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_expectation(expectation)
cutn.destroy_network_operator(hamiltonian)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Freed memory on GPU")
print("Finalized the cuTensorNet library")
