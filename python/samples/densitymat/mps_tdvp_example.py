# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MPS-TDVP time propagation example (pythonic API).

Demonstrates propagating a pure MPS state under a transverse-field Ising
Hamiltonian encoded as an MPO, using the split-scope TDVP method with
Krylov subspace exponentiation.

Workflow:
 1. Build Hamiltonian as MPO (nearest-neighbor ZZ + transverse X field)
 2. Create input and output MPS states
 3. Initialize input MPS to a Neel state |0101...>
 4. Create TimePropagation object with TDVP + Krylov configs
 5. Prepare and run time-stepping loop
"""

import time

import cupy as cp
import numpy as np

from cuquantum.densitymat import (
    MatrixProductOperator,
    MPSPureState,
    Operator,
    WorkStream,
    TimePropagation,
    TDVPConfig,
    KrylovConfig,
    mpo_product,
)

# ---------- Simulation parameters ----------
NUM_SITES    = 40
PHYS_DIM     = 2
MAX_BOND_DIM = 128
MPO_BOND_DIM = 3
NUM_STEPS    = 3
DT           = 0.01
J_COUPLING   = 1.0
H_FIELD      = 0.5
DTYPE        = "complex128"
NP_DTYPE     = np.complex128
VERBOSE      = True


# ============================================================================
# Transverse-field Ising MPO builder
# ============================================================================
#
# H = -J * sum_i Z_i Z_{i+1}  +  h * sum_i X_i
#
# Standard MPO representation with bond dimension 3.
# Bulk MPO matrix (indexed by bond dims aL, aR):
#
#        |  I      0     0  |
#   W =  |  Z      0     0  |
#        | h*X   -J*Z    I  |
#
# Left boundary:  W_L = [ h*X,  -J*Z,  I ]
# Right boundary: W_R = [ I;  Z;  h*X ]^T

def build_ising_mpo():
    """Build transverse-field Ising MPO tensors on GPU (column-major layout)."""
    I2 = np.eye(2, dtype=NP_DTYPE)
    X2 = np.array([[0, 1], [1, 0]], dtype=NP_DTYPE)
    Z2 = np.array([[1, 0], [0, -1]], dtype=NP_DTYPE)
    jc = -J_COUPLING
    hc = H_FIELD

    gpu_tensors = []
    for site in range(NUM_SITES):
        bL = 1 if site == 0 else MPO_BOND_DIM
        bR = 1 if site == NUM_SITES - 1 else MPO_BOND_DIM
        T = np.zeros((bL, PHYS_DIM, bR, PHYS_DIM), dtype=NP_DTYPE, order='F')

        if NUM_SITES == 1:
            T[0, :, 0, :] = hc * X2
        elif site == 0:
            T[0, :, 0, :] = hc * X2
            T[0, :, 1, :] = jc * Z2
            T[0, :, 2, :] = I2
        elif site == NUM_SITES - 1:
            T[0, :, 0, :] = I2
            T[1, :, 0, :] = Z2
            T[2, :, 0, :] = hc * X2
        else:
            T[0, :, 0, :] = I2
            T[1, :, 0, :] = Z2
            T[2, :, 0, :] = hc * X2
            T[2, :, 1, :] = jc * Z2
            T[2, :, 2, :] = I2

        gpu_tensors.append(cp.asfortranarray(cp.asarray(T)))

    return gpu_tensors


# ============================================================================
# Build a Neel-state MPS  |0,1,0,1,...>
# ============================================================================

def build_neel_mps(bond_dims):
    """Build Neel state MPS tensors on GPU (column-major layout)."""
    gpu_tensors = []
    for site in range(NUM_SITES):
        bL = 1 if site == 0 else bond_dims[site - 1]
        bR = 1 if site == NUM_SITES - 1 else bond_dims[site]
        T = np.zeros((bL, PHYS_DIM, bR), dtype=NP_DTYPE, order='F')
        neel_spin = site % 2
        T[0, neel_spin, 0] = 1.0
        gpu_tensors.append(cp.asfortranarray(cp.asarray(T)))
    return gpu_tensors


# ============================================================================
# Main workflow
# ============================================================================

def main():
    space_shape = (PHYS_DIM,) * NUM_SITES
    mpo_bond_dims = (MPO_BOND_DIM,) * (NUM_SITES - 1)

    # --- 1. Build transverse-field Ising MPO ---
    mpo_tensors = build_ising_mpo()
    mpo = MatrixProductOperator(mpo_tensors, space_shape, mpo_bond_dims)
    if VERBOSE:
        print(f"Built transverse-field Ising MPO (bond dim {MPO_BOND_DIM})")

    # --- 2. Build Hamiltonian operator from MPO ---
    term = mpo_product((mpo, list(range(NUM_SITES))))
    hamiltonian = Operator(space_shape, (term,))
    if VERBOSE:
        print("Constructed Hamiltonian operator from MPO")

    # --- 3. Create input and output MPS states ---
    mps_bond_dims = []
    for i in range(NUM_SITES - 1):
        left_dim = int(np.prod(space_shape[:i + 1]))
        right_dim = int(np.prod(space_shape[i + 1:]))
        mps_bond_dims.append(min(MAX_BOND_DIM, left_dim, right_dim))
    mps_bond_dims = tuple(mps_bond_dims)

    # --- 3a. Allocate GPU storage for MPS tensors before querying free memory ---
    neel_tensors = build_neel_mps(mps_bond_dims)
    state_out_bufs = [cp.zeros_like(t) for t in neel_tensors]

    free_mem, _ = cp.cuda.Device().mem_info
    free_mem = int(free_mem * 0.95)
    if VERBOSE:
        print(f"Available workspace memory (bytes) = {free_mem}")

    ctx = WorkStream(memory_limit=free_mem)

    batch_size = 1
    state_in = MPSPureState(ctx, space_shape, mps_bond_dims, batch_size, DTYPE)
    state_out = MPSPureState(ctx, space_shape, mps_bond_dims, batch_size, DTYPE)

    if VERBOSE:
        print(f"MPS state has {NUM_SITES} site tensors, bond dims: {mps_bond_dims}")

    # --- 4. Attach storage ---
    state_in.attach_storage(neel_tensors)
    state_out.attach_storage(state_out_bufs)

    if VERBOSE:
        print("Initialized MPS states (Neel state |0101...>)")

    # --- 5. Create TDVP time propagation object ---
    tdvp_config = TDVPConfig(order=2)
    krylov_config = KrylovConfig(tolerance=1e-8, max_dim=10)

    propagator = TimePropagation(
        hamiltonian,
        is_hermitian=True,
        scope="split",
        approach="krylov",
        scope_config=tdvp_config,
        approach_config=krylov_config,
    )
    if VERBOSE:
        print("Created TDVP time propagation object")
        print("Configured TDVP (order=2, 1-site) with Krylov (max_dim=10, tol=1e-8)")

    # --- 6. Prepare propagation ---
    propagator.prepare(ctx, state_in, state_out)
    if VERBOSE:
        print("Prepared time propagation")

    # --- 7. Time-stepping loop ---
    if VERBOSE:
        print(f"\nStarting TDVP propagation: {NUM_STEPS} steps, dt = {DT}")
        print(f"Hamiltonian: H = -{J_COUPLING} sum Z_i Z_{{i+1}} + {H_FIELD} sum X_i\n")

    cp.cuda.Device().synchronize()
    wall_start = time.perf_counter()

    for step in range(NUM_STEPS):
        current_time = step * DT
        propagator.compute(DT, current_time, None, state_in, state_out)
        state_in, state_out = state_out, state_in

    cp.cuda.Device().synchronize()
    wall_end = time.perf_counter()
    if VERBOSE:
        print(f"\nTotal propagation wall time (sec) = {wall_end - wall_start:.6f}")

    if VERBOSE:
        print("Finished computation and exit.")


if __name__ == "__main__":
    main()
