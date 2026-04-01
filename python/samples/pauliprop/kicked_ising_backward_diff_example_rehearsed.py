#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Reverse-mode AD of the IBM 127-qubit kicked Ising experiment with rehearsal.

An example demonstrating use of cuPauliProp's automatic differentiation facilities for
calculating expectation gradients of parameterised quantum circuits, with rehearsal-based
memory allocation. We use the circuit of IBM's 127-qubit kicked Ising experiment, as
presented in Nature volume 618, pages 500-505 (2023), without simulated noise or twirling.
We consider observable Z_62 and the circuit of 20 Trotter steps, as correspond to Fig 4. b),
and compute its expectation gradient at the point where all X rotations have angle PI/4 and
all ZZ rotations are -PI/2.

Instead of relying on auto-allocation during computation, we first perform a full rehearsal
of every operation (forward gate applications, trace, backward trace, undo gates, and
backward gate applications) to determine the maximum required buffer capacities and
workspace sizes. We then preallocate exact-sized buffers and a fixed workspace allocator,
eliminating allocation overhead during the timed computation.

The backward differentiation uses O(1) memory in the number of circuit layers by replaying
the computation tape in reverse rather than caching intermediates.
"""

from typing import Sequence

import cupy as cp
import numpy as np
from cuquantum import MemoryPointer

from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)


class PreAllocatedWorkspaceAllocator:
    """
    A simple allocator that always returns the same preallocated block.

    This allocator preallocates a contiguous block of GPU memory and returns
    the same pointer for every allocation request (as long as size fits).

    Warning:
        This allocator is "unsafe" because it does not track whether the
        memory is currently in use. It is the caller's responsibility to
        ensure that allocations do not overlap in time.

    Args:
        device_id: The GPU device ID.
        total_size: Total size in bytes of the preallocated block.
    """

    def __init__(self, device_id: int, total_size: int):
        self.device_id = device_id
        self.total_size = total_size

        with cp.cuda.Device(device_id):
            self._buffer = cp.empty(total_size, dtype=cp.uint8)
        self._ptr = self._buffer.data.ptr

    def memalloc_async(self, size: int, stream) -> MemoryPointer:
        """Return a pointer to the preallocated block."""
        if size > self.total_size:
            raise MemoryError(
                f"PreallocatedWorkspaceAllocator: requested {size} bytes, "
                f"but only {self.total_size} preallocated"
            )
        return MemoryPointer(self._ptr, size, finalizer=None)


NUM_CIRCUIT_QUBITS = 127
NUM_ROTATIONS_PER_LAYER = 48
PI = np.pi
TARGET_MAX_TERMS = 250_000

ZZ_QUBITS_RED = np.array([
    [  2,   1],  [ 33,  39], [ 59,  60], [ 66,  67], [ 72,  81], [118, 119],
    [ 21,  20],  [ 26,  25], [ 13,  12], [ 31,  32], [ 70,  74], [122, 123],
    [ 96,  97],  [ 57,  56], [ 63,  64], [107, 108], [103, 104], [ 46,  45],
    [ 28,  35],  [  7,   6], [ 79,  78], [  5,   4], [109, 114], [ 62,  61],
    [ 58,  71],  [ 37,  52], [ 76,  77], [  0,  14], [ 36,  51], [106, 105],
    [ 73,  85],  [ 88,  87], [ 68,  55], [116, 115], [ 94,  95], [100, 110],
    [ 17,  30],  [ 92, 102], [ 50,  49], [ 83,  84], [ 48,  47], [ 98,  99],
    [  8,   9],  [121, 120], [ 23,  24], [ 44,  43], [ 22,  15], [ 53,  41]
], dtype=np.int32)

ZZ_QUBITS_BLUE = np.array([
    [ 53,  60], [123, 124], [ 21,  22], [ 11,  12], [ 67,  68], [  2,   3],
    [ 66,  65], [122, 121], [110, 118], [  6,   5], [ 94,  90], [ 28,  29],
    [ 14,  18], [ 63,  62], [111, 104], [100,  99], [ 45,  44], [  4,  15],
    [ 20,  19], [ 57,  58], [ 77,  71], [ 76,  75], [ 26,  27], [ 16,   8],
    [ 35,  47], [ 31,  30], [ 48,  49], [ 69,  70], [125, 126], [ 89,  74],
    [ 80,  79], [116, 117], [114, 113], [ 10,   9], [106,  93], [101, 102],
    [ 92,  83], [ 98,  91], [ 82,  81], [ 54,  64], [ 96, 109], [ 85,  84],
    [ 87,  86], [108, 112], [ 34,  24], [ 42,  43], [ 40,  41], [ 39,  38]
], dtype=np.int32)

ZZ_QUBITS_GREEN = np.array([
    [ 10,  11], [ 54,  45], [111, 122], [ 64,  65], [ 60,  61], [103, 102],
    [ 72,  62], [  4,   3], [ 33,  20], [ 58,  59], [ 26,  16], [ 28,  27],
    [  8,   7], [104, 105], [ 73,  66], [ 87,  93], [ 85,  86], [ 55,  49],
    [ 68,  69], [ 89,  88], [ 80,  81], [117, 118], [101, 100], [114, 115],
    [ 96,  95], [ 29,  30], [106, 107], [ 83,  82], [ 91,  79], [  0,   1],
    [ 56,  52], [ 90,  75], [126, 112], [ 36,  32], [ 46,  47], [ 77,  78],
    [ 97,  98], [ 17,  12], [119, 120], [ 22,  23], [ 24,  25], [ 43,  34],
    [ 42,  41], [ 40,  39], [ 37,  38], [125, 124], [ 50,  51], [ 18,  19]
], dtype=np.int32)


def get_pauli_string_as_packed_integers(paulis: Sequence[str], qubits: Sequence[int], num_qubits: int) -> np.ndarray:
    num_packed_ints = get_num_packed_integers(num_qubits)
    out = np.zeros(num_packed_ints * 2, dtype=np.uint64)
    x_ptr = out[:num_packed_ints]
    z_ptr = out[num_packed_ints:]
    for pauli, qubit in zip(paulis, qubits):
        int_ind = qubit // 64
        bit_ind = qubit % 64
        if pauli in ("X", "Y"):
            x_ptr[int_ind] |= 1 << bit_ind
        if pauli in ("Z", "Y"):
            z_ptr[int_ind] |= 1 << bit_ind
    return out


def get_x_rotation_layer(angle: float) -> list[PauliRotationGate]:
    return [PauliRotationGate(angle, ["X"], [i]) for i in range(NUM_CIRCUIT_QUBITS)]


def get_zz_rotation_layer(angle: float, topology: np.ndarray) -> list[PauliRotationGate]:
    return [PauliRotationGate(angle, ["Z", "Z"], pair.tolist()) for pair in topology]


def get_ibm_heavy_hex_ising_circuit(x_angle: float, zz_angle: float, num_trotter_steps: int) -> list[PauliRotationGate]:
    circuit: list[PauliRotationGate] = []
    for _ in range(num_trotter_steps):
        circuit.extend(get_x_rotation_layer(x_angle))
        circuit.extend(get_zz_rotation_layer(zz_angle, ZZ_QUBITS_RED))
        circuit.extend(get_zz_rotation_layer(zz_angle, ZZ_QUBITS_BLUE))
        circuit.extend(get_zz_rotation_layer(zz_angle, ZZ_QUBITS_GREEN))
    return circuit


def main():
    print("cuPauliProp Kicked Ising — Reverse-Mode AD Example (rehearsed)")
    print("=" * 70)
    print()

    handle = LibraryHandle()

    # Step 1: Create an empty expansion for rehearsal (uses minimal internal buffers).
    options = PauliExpansionOptions(memory_limit="80%", blocking=True)
    expansion = PauliExpansion.empty(
        handle,
        NUM_CIRCUIT_QUBITS,
        TARGET_MAX_TERMS,
        dtype="float64",
        sort_order=None,
        has_duplicates=True,
        options=options,
    )

    # Step 2: Define truncation strategy.
    truncation = Truncation(pauli_coeff_cutoff=1e-4, pauli_weight_cutoff=8)

    # Step 3: Build the quantum circuit (kicked Ising model).
    x_angle = PI / 4.0
    zz_angle = -PI / 2.0
    num_trotter_steps = 20
    circuit = get_ibm_heavy_hex_ising_circuit(x_angle, zz_angle, num_trotter_steps)
    num_gates = len(circuit)

    print(f"Circuit: 127-qubit IBM heavy-hex Ising")
    print(f"  Trotter steps: {num_trotter_steps}")
    print(f"  Total gates:   {num_gates}")
    print(f"  X angle:       {x_angle:.4f} (pi/4)")
    print(f"  ZZ angle:      {zz_angle:.4f} (-pi/2)")
    print()

    # Step 4: Rehearse ALL operations to find max buffer and workspace requirements.
    #
    # The full backward-diff computation consists of:
    #   Forward:  apply_gate(adjoint=True) for each gate in reverse order
    #   Trace:    trace_with_zero_state
    #   Bwd trace: trace_with_zero_state_backward_diff
    #   Backward: for each gate in forward order:
    #     a) apply_gate(adjoint=False)           -- undo forward gate
    #     b) apply_gate_backward_diff(adjoint=True)   -- backward through gate
    #
    # Because the different operations return different RehearsalInfo
    # subclasses (GateApplicationRehearsalInfo, TraceBackwardRehearsalInfo,
    # and the base RehearsalInfo), we keep separate accumulators for each
    # type and combine them manually at the end.  Mixing subclasses via |
    # would decay to the base class, losing the subclass-specific fields
    # (num_terms_required, cotangent_num_terms) that we need for sizing.

    print("Rehearsing ...")

    # Accumulator for apply_gate and apply_gate_backward_diff (GateApplicationRehearsalInfo)
    max_gate_info = None

    # 4a: Rehearse forward pass gates (adjoint=True)
    for gate_index in range(num_gates - 1, -1, -1):
        gate = circuit[gate_index]
        info = expansion.apply_gate(
            gate, truncation=truncation, adjoint=True,
            sort_order=None, keep_duplicates=False,
        )
        max_gate_info = max_gate_info | info if max_gate_info is not None else info

    # 4b: Rehearse trace_with_zero_state (returns base RehearsalInfo)
    trace_info = expansion.trace_with_zero_state(rehearse=True)

    # 4c: Rehearse trace_with_zero_state_backward_diff (returns TraceBackwardRehearsalInfo;
    #     cotangent values are unused during rehearsal, pass dummy scalars)
    bwd_trace_info = expansion.trace_with_zero_state_backward_diff(0.0, 0.0, rehearse=True)

    # 4d: Rehearse backward pass: undo gates (adjoint=False) and apply_gate_backward_diff
    for gate_index in range(num_gates):
        gate = circuit[gate_index]

        # Undo gate (adjoint=False)
        undo_info = expansion.apply_gate(
            gate, truncation=truncation, adjoint=False,
            sort_order=None, keep_duplicates=False,
        )
        max_gate_info = max_gate_info | undo_info

        # Backward through gate (adjoint=True)
        bwd_gate_info = expansion.apply_gate_backward_diff(
            gate=gate,
            cotangent_out=expansion.view(),
            truncation=truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
            rehearse=True,
        )
        max_gate_info = max_gate_info | bwd_gate_info

    # Combine: the buffer capacity must accommodate both the expansion
    # terms from gate applications and the cotangent terms from the
    # backward trace.
    capacity = max(max_gate_info.num_terms_required, bwd_trace_info.cotangent_num_terms)
    max_workspace = max(
        max_gate_info.device_scratch_workspace_required,
        trace_info.device_scratch_workspace_required,
        bwd_trace_info.device_scratch_workspace_required,
    )
    max_host_workspace = max(
        max_gate_info.host_scratch_workspace_required,
        trace_info.host_scratch_workspace_required,
        bwd_trace_info.host_scratch_workspace_required,
    )

    print(f"  Rehearsal complete.")
    print(f"  Gate rehearsal:           {max_gate_info}")
    print(f"  Backward trace rehearsal: {bwd_trace_info}")
    print(f"  Combined capacity:        {capacity:,} terms")
    print(f"  Combined device workspace: {max_workspace:,} bytes")
    print()

    # Step 5: Allocate buffers at max required capacity based on rehearsal.
    ints_per_string = get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    xz_shape = (capacity, 2 * ints_per_string)
    coef_shape = (capacity,)

    in_xz = cp.zeros(xz_shape, dtype=cp.uint64, order="C")
    in_coef = cp.zeros(coef_shape, dtype=cp.float64)
    out_xz = cp.empty_like(in_xz)
    out_coef = cp.empty_like(in_coef)
    cot_in_xz = cp.empty_like(in_xz)
    cot_in_coef = cp.empty_like(in_coef)
    cot_out_xz = cp.empty_like(in_xz)
    cot_out_coef = cp.empty_like(in_coef)

    # Step 6: Encode the observable (Z_62) into the input buffer.
    in_xz[0] = cp.asarray(get_pauli_string_as_packed_integers(["Z"], [62], NUM_CIRCUIT_QUBITS))
    in_coef[0] = 1.0

    # Step 7: Create a preallocated workspace allocator based on rehearsed requirements.
    workspace_bytes = max_workspace
    workspace_allocator = (
        PreAllocatedWorkspaceAllocator(device_id=0, total_size=workspace_bytes)
        if workspace_bytes > 0 else None
    )

    options_sized = PauliExpansionOptions(
        allocator=workspace_allocator,
        memory_limit=workspace_bytes or "80%",
        blocking=True,
    )

    # Step 8: Create real expansions from buffers for ping-pong pattern.
    in_expansion = expansion.from_empty(
        in_xz, in_coef,
        num_terms=1,
        sort_order="little_endian_bitwise",
        has_duplicates=False,
        options=options_sized,
    )

    out_expansion = PauliExpansion(
        handle, NUM_CIRCUIT_QUBITS, 0,
        out_xz, out_coef,
        sort_order=None, has_duplicates=True,
        options=options_sized,
    )

    cot_in_expansion = PauliExpansion(
        handle, NUM_CIRCUIT_QUBITS, 0,
        cot_in_xz, cot_in_coef,
        sort_order=None, has_duplicates=True,
        options=options_sized,
    )

    cot_out_expansion = PauliExpansion(
        handle, NUM_CIRCUIT_QUBITS, 0,
        cot_out_xz, cot_out_coef,
        sort_order=None, has_duplicates=True,
        options=options_sized,
    )

    total_buffer_bytes = 4 * (in_xz.nbytes + in_coef.nbytes) + (workspace_bytes or 0)
    print(f"Allocated {total_buffer_bytes:,} bytes ({total_buffer_bytes / (1 << 20):.1f} MiB)")
    print(f"  Expansion capacity: {capacity:,} terms (x4 expansions)")
    print(f"  Workspace:          {workspace_bytes:,} bytes")
    print()

    # ================================================================
    # Step 9: Forward pass — back-propagate observable through adjoint circuit.
    # ================================================================
    print("Forward pass ...")
    fwd_start = cp.cuda.Event()
    fwd_end = cp.cuda.Event()
    fwd_start.record()
    max_num_terms = 1

    for gate_index in range(num_gates - 1, -1, -1):
        gate = circuit[gate_index]
        in_expansion.apply_gate(
            gate, truncation=truncation,
            expansion_out=out_expansion,
            adjoint=True, keep_duplicates=False,
        )
        max_num_terms = max(max_num_terms, out_expansion.num_terms)
        in_expansion, out_expansion = out_expansion, in_expansion

        if in_expansion.num_terms > TARGET_MAX_TERMS:
            raise RuntimeError(
                f"num_terms exceeded TARGET_MAX_TERMS during forward pass: "
                f"{in_expansion.num_terms} > {TARGET_MAX_TERMS}"
            )

    # Step 10: Compute trace (expectation value).
    trace_significand, trace_exponent = in_expansion.trace_with_zero_state()
    expec = trace_significand * np.exp2(trace_exponent)

    fwd_end.record()
    fwd_end.synchronize()
    fwd_time = cp.cuda.get_elapsed_time(fwd_start, fwd_end) / 1000.0

    print(f"  Expectation value: {expec}")
    print(f"  Final num terms:   {in_expansion.num_terms}")
    print(f"  Forward time (s):  {fwd_time:.3f}")
    print()

    # ================================================================
    # Step 11: Backward pass — reverse-mode AD with tape replay.
    # ================================================================
    print("Backward pass ...")
    bwd_start = cp.cuda.Event()
    bwd_start.record()

    # Seed: backward through trace_with_zero_state.
    upstream_seed = 1.0
    cotangent_trace_significand = upstream_seed * np.exp2(trace_exponent)
    cotangent_trace_exponent = (
        upstream_seed
        * trace_significand
        * np.exp2(trace_exponent)
        * np.log(2.0)
    )
    cot_out_expansion = in_expansion.trace_with_zero_state_backward_diff(
        cotangent_trace_significand,
        cotangent_trace_exponent,
        cotangent_expansion=cot_out_expansion,
    )

    grad_x_angle = 0.0
    grad_zz_angle = 0.0

    for gate_index in range(num_gates):
        gate = circuit[gate_index]

        # Step a: undo the forward gate to recover the previous intermediate.
        in_expansion.apply_gate(
            gate, truncation=truncation,
            expansion_out=out_expansion,
            adjoint=False, keep_duplicates=False,
        )
        in_expansion, out_expansion = out_expansion, in_expansion

        # Step b: backward through this gate application.
        cot_in_expansion, param_grads = in_expansion.apply_gate_backward_diff(
            gate=gate,
            cotangent_out=cot_out_expansion.view(),
            truncation=truncation,
            cotangent_in=cot_in_expansion,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )

        if param_grads is not None:
            if len(gate.pauli_string) == 1 and gate.pauli_string[0] == "X":
                grad_x_angle += param_grads[0]
            else:
                grad_zz_angle += param_grads[0]

        cot_out_expansion, cot_in_expansion = cot_in_expansion, cot_out_expansion

    bwd_end = cp.cuda.Event()
    bwd_end.record()
    bwd_end.synchronize()
    bwd_time = cp.cuda.get_elapsed_time(bwd_start, bwd_end) / 1000.0

    print(f"  d<Z_62>/d(x_angle):  {grad_x_angle}")
    print(f"  d<Z_62>/d(zz_angle): {grad_zz_angle}")
    print(f"  Backward time (s):   {bwd_time:.3f}")
    print()

    # Step 12: Summary.
    print(f"Rehearsed requirements:")
    print(f"  {max_gate_info}")
    print(f"  {bwd_trace_info}")
    print()


if __name__ == "__main__":
    main()
