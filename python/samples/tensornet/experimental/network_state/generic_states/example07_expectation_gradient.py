# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Expectation value with gradient computation using NetworkState.

This example shows how to compute gradients of a scalar loss from expectation values:
  1. Build a state by applying tensor operators; mark those you want gradients for with gradient=True.
  2. Define the observable (e.g. a Hamiltonian as Pauli strings).
  3. Compute expectation value E and define loss L = E^2.
  4. Call compute_expectation_with_gradients with expectation_value_adjoint=dL/dE=2E to get
     gradients of L with respect to the parameters of the gradient-enabled operators.

  Circuit (4 qubits):
    q0: RY(pi/4)[gradient], CNOT(0,1)
    q1: H, CNOT(1,2), RX(pi/4)[gradient]
    q2: CNOT(2,3), RZ(pi/4)[gradient]
    q3: H

  Hamiltonian: 2*X_0*Y_1*Z_2*Z_3 + 3*Z_1*Z_2 + 5*Z_0*Y_2*Y_3
"""

import cmath
import math

import cupy as cp
import numpy as np

from cuquantum.tensornet.experimental import NetworkState

num_qubits = 4
state_mode_extents = (2,) * num_qubits
dtype = "complex128"

# Gate matrices (2x2)
inv_sqrt2 = 1.0 / (2.0 ** 0.5)
h_gate = cp.array([[1, 1], [1, -1]], dtype=dtype) * inv_sqrt2

theta = math.pi / 4.0
cy, sy = math.cos(theta / 2), math.sin(theta / 2)
ry_gate = cp.array([[cy, -sy], [sy, cy]], dtype=dtype)

c = math.cos(theta / 2)
s = -1j * math.sin(theta / 2)
rx_gate = cp.array([[c, s], [s, c]], dtype=dtype)


rz_00 = cmath.exp(-1j * theta / 2)
rz_11 = cmath.exp(1j * theta / 2)
rz_gate = cp.array([[rz_00, 0.0], [0.0, rz_11]], dtype=dtype)

# CNOT: control first mode, target second mode (2,2,2,2)
cx_gate = cp.zeros((2, 2, 2, 2), dtype=dtype)
cx_gate[0, 0, 0, 0] = 1.0
cx_gate[0, 1, 0, 1] = 1.0
cx_gate[1, 0, 1, 1] = 1.0
cx_gate[1, 1, 1, 0] = 1.0

# Hamiltonian: 2*X_0*Y_1*Z_2*Z_3 + 3*Z_1*Z_2 + 5*Z_0*I_1*Y_2*Y_3 (4-qubit Pauli strings)
hamiltonian = {"XYZZ": 2.0, "IZZI": 3.0, "ZIYY": 5.0}


with NetworkState(state_mode_extents, dtype=dtype) as state:
    
    # Build state
    state.apply_tensor_operator((1,), h_gate, unitary=True)
    state.apply_tensor_operator((3,), h_gate, unitary=True)
    state.apply_tensor_operator((1, 2), cx_gate, unitary=True)
    state.apply_tensor_operator((2, 3), cx_gate, unitary=True)

    ry_tensor_id = state.apply_tensor_operator((0,), ry_gate, unitary=True, gradient=True)
    rx_tensor_id = state.apply_tensor_operator((1,), rx_gate, unitary=True, gradient=True)
    rz_tensor_id = state.apply_tensor_operator((2,), rz_gate, unitary=True, gradient=True)

    state.apply_tensor_operator((0, 1), cx_gate, unitary=True)

    # Define loss L = real(E)^2. By chain rule:
    #   dL/dtheta = dL/dE * dE/dtheta,  with  dL/dE = 2*real(E).
    # compute_expectation_with_gradients computes dE/dtheta scaled by expectation_value_adjoint,
    # so we pass dL/dE = 2*real(E) as expectation_value_adjoint to get dL/dtheta directly.
    # Since dL/dE depends on E, we must compute E first.
    expectation_value = state.compute_expectation(hamiltonian)
    ex = expectation_value.real if hasattr(expectation_value, "real") else expectation_value
    loss = ex * ex

    expectation_value_adjoint = 2.0 * ex  # dL/dE
    _, _, gradients = state.compute_expectation_with_gradients(
        hamiltonian,
        expectation_value_adjoint=expectation_value_adjoint,
    )

print(f"Expectation value (real): {ex:.4f}")
print(f"Loss (E^2): {loss:.4f}")
print(f"dLoss/dExpectation: {expectation_value_adjoint:.4f}")

# Gradients are returned as a dict: tensor_id -> gradient array (same shape as the gate)
loss_grad_ry = gradients[ry_tensor_id].get()
loss_grad_rx = gradients[rx_tensor_id].get()
loss_grad_rz = gradients[rz_tensor_id].get()

print(f"dLoss/dRY gate (q0, tensor_id={ry_tensor_id}):\n{loss_grad_ry}")
print(f"dLoss/dRX gate (q1, tensor_id={rx_tensor_id}):\n{loss_grad_rx}")
print(f"dLoss/dRZ gate (q2, tensor_id={rz_tensor_id}):\n{loss_grad_rz}")

# Gradient with respect to the scalar rotation angle theta for each gate.
# The cotangents G = dL/dU* returned above are Wirtinger-style derivatives w.r.t. the
# complex matrix entries. To recover the real-parameter gradient, apply the chain rule:
#   dL/dtheta = 2 * Re(sum_{i,j} conj(G_ij) * dU_ij/dtheta)
#             = 2 * Re(<G, dU/dtheta>_F)   (Frobenius inner product, np.vdot conjugates first arg)
#
# Analytic derivatives of each gate matrix w.r.t. theta:
#   dRY/dtheta[i,j]:  [[-sy/2, -cy/2], [ cy/2, -sy/2]]
#   dRX/dtheta[i,j]:  [[-sy/2, -i*cy/2], [-i*cy/2, -sy/2]]   (s = -i*sy, so ds/dtheta = -i*cy/2)
#   dRZ/dtheta[i,j]:  [[-i/2 * rz_00, 0], [0, i/2 * rz_11]]
dU_ry_dtheta = np.array([[-sy / 2, -cy / 2], [cy / 2, -sy / 2]])
dU_rx_dtheta = np.array([[-sy / 2, -1j * cy / 2], [-1j * cy / 2, -sy / 2]])
dU_rz_dtheta = np.array([[-1j / 2 * rz_00, 0.0], [0.0, 1j / 2 * rz_11]])

dl_dtheta_ry = 2.0 * np.real(np.vdot(loss_grad_ry, dU_ry_dtheta))
dl_dtheta_rx = 2.0 * np.real(np.vdot(loss_grad_rx, dU_rx_dtheta))
dl_dtheta_rz = 2.0 * np.real(np.vdot(loss_grad_rz, dU_rz_dtheta))

print(f"\ndL/dtheta_RY (q0): {dl_dtheta_ry:.4f}")
print(f"dL/dtheta_RX (q1): {dl_dtheta_rx:.4f}")
print(f"dL/dtheta_RZ (q2): {dl_dtheta_rz:.4f}")
