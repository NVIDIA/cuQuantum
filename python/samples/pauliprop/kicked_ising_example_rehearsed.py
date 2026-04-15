#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: IBM 127-qubit Kicked Ising simulation using the pythonic cuPauliProp API with rehearsal.

This ports the bindings sample in ``samples/bindings/cupauliprop/kicked_ising_example.py``
to the higher-level pythonic API. It back-propagates the Z_62 observable through the
127-qubit heavy-hex Ising circuit (20 Trotter steps, X angle pi/4) and evaluates the
expectation value with respect to initial state |0...0>.
It uses a preallocated workspace allocator to avoid repeated overhead from memory pool fragmentation.
"""
# Sphinx
import time
from typing import Sequence

import cupy as cp
import numpy as np
from cuquantum import MemoryPointer

from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    RehearsalInfo,
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


# Circuit constants (IBM heavy-hex kicked Ising)
NUM_CIRCUIT_QUBITS = 127
NUM_ROTATIONS_PER_LAYER = 48
PI = np.pi
ZZ_ROTATION_ANGLE = -PI / 2.0
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


def get_zz_rotation_layer(topology: np.ndarray) -> list[PauliRotationGate]:
    return [PauliRotationGate(ZZ_ROTATION_ANGLE, ["Z", "Z"], pair.tolist()) for pair in topology]


def get_ibm_heavy_hex_ising_circuit(x_rotation_angle: float, num_trotter_steps: int) -> list[PauliRotationGate]:
    circuit: list[PauliRotationGate] = []
    for _ in range(num_trotter_steps):
        circuit.extend(get_x_rotation_layer(x_rotation_angle))
        circuit.extend(get_zz_rotation_layer(ZZ_QUBITS_RED))
        circuit.extend(get_zz_rotation_layer(ZZ_QUBITS_BLUE))
        circuit.extend(get_zz_rotation_layer(ZZ_QUBITS_GREEN))
    return circuit


def main():
    print("cuPauliProp IBM Heavy-hex Ising Example (pythonic API, rehearsed)")
    print("=" * 70)
    print()

    # Step 1: Create a library handle.
    handle = LibraryHandle()

    # Step 2: Create an empty expansion for rehearsal (uses minimal internal buffers).
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

    # Step 3: Define truncation strategy.
    truncation = Truncation(pauli_coeff_cutoff=1e-4, pauli_weight_cutoff=8)
    num_gates_between_truncations = 10

    # Step 4: Build the quantum circuit (kicked Ising model).
    x_rotation_angle = PI / 4.0
    num_trotter_steps = 20
    circuit = get_ibm_heavy_hex_ising_circuit(x_rotation_angle, num_trotter_steps)

    print(f"Circuit: 127-qubit IBM heavy-hex Ising")
    print(f"  Trotter steps: {num_trotter_steps}")
    print(f"  Total gates:   {len(circuit)}")
    print(f"  Rx angle:      {x_rotation_angle} (pi/4)")
    print()

    # Step 5: Rehearse all gate applications to find max terms and workspace requirements.
    max_rehearsal_info = None
    for gate_index in range(len(circuit) - 1, -1, -1):
        gate = circuit[gate_index]
        active_truncation = truncation if (gate_index % num_gates_between_truncations == 0) else None
        rehearsal_info = expansion.apply_gate(
            gate,
            truncation=active_truncation,
            adjoint=True,
            keep_duplicates=False,
        )
        max_rehearsal_info = max_rehearsal_info | rehearsal_info if max_rehearsal_info is not None else rehearsal_info

    # Step 6: Rehearse the trace operation as well.
    trace_rehearsal_info = expansion.trace_with_zero_state(rehearse=True)
    max_rehearsal_info = max_rehearsal_info | trace_rehearsal_info

    # Step 7: Allocate buffers at max required capacity based on rehearsal.
    ints_per_string = get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    capacity = max_rehearsal_info.num_terms_required
    xz_shape = (capacity, 2 * ints_per_string)
    coef_shape = (capacity,)

    in_xz = cp.zeros(xz_shape, dtype=cp.uint64, order="C")
    in_coef = cp.zeros(coef_shape, dtype=cp.float64)
    out_xz = cp.empty_like(in_xz)
    out_coef = cp.empty_like(in_coef)

    # Step 8: Encode the observable (Z_62) into the input buffer.
    in_xz[0] = cp.asarray(get_pauli_string_as_packed_integers(["Z"], [62], NUM_CIRCUIT_QUBITS))
    in_coef[0] = 1.0

    # Step 9: Create a preallocated workspace allocator based on rehearsed requirements.
    workspace_bytes = max_rehearsal_info.device_scratch_workspace_required
    workspace_allocator = (
        PreAllocatedWorkspaceAllocator(device_id=0, total_size=workspace_bytes)
        if workspace_bytes > 0 else None
    )
        
    options_sized = PauliExpansionOptions(
        allocator=workspace_allocator,
        memory_limit=workspace_bytes or "80%",
        blocking=True,
    )

    # Step 10: Create real expansions from buffers for ping-pong pattern.
    # Use sort_order="little_endian_bitwise" because a Pauli expansion with a single term is always sorted.
    in_expansion = expansion.from_empty(
        in_xz, in_coef,
        num_terms=1,
        sort_order="little_endian_bitwise",
        has_duplicates=False,
        options=options_sized,
    )

    out_expansion = PauliExpansion(
        handle,
        NUM_CIRCUIT_QUBITS,
        0,
        out_xz,
        out_coef,
        sort_order=None,
        has_duplicates=True,
        options=options_sized,
    )

    # Step 11: Back-propagate through the adjoint circuit.
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    max_num_terms = 1

    for gate_index in range(len(circuit) - 1, -1, -1):
        gate = circuit[gate_index]
        active_truncation = truncation if (gate_index % num_gates_between_truncations == 0) else None
        in_expansion.apply_gate(
            gate,
            truncation=active_truncation,
            expansion_out=out_expansion,
            adjoint=True,
            keep_duplicates=False,
        )
        # Track maximum output term count after each gate.
        max_num_terms = max(max_num_terms, out_expansion.num_terms)
        
        # Swap input and output expansions (ping-pong).
        in_expansion, out_expansion = out_expansion, in_expansion

        # Enforce runtime bound on valid terms (buffers may be larger).
        if in_expansion.num_terms > TARGET_MAX_TERMS:
            raise RuntimeError(
                f"num_terms exceeded TARGET_MAX_TERMS during compute: "
                f"{in_expansion.num_terms} > {TARGET_MAX_TERMS}"
            )

    # After the loop, in_expansion holds the final result (from the last swap). Release out_expansion
    out_expansion = None
    # Step 12: Compute the expectation value <Z_62> with respect to |0...0>.
    trace_significand, trace_exponent = in_expansion.trace_with_zero_state()
    expec = trace_significand * np.exp2(trace_exponent)

    end_event.record()
    end_event.synchronize()
    duration = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0  # seconds

    # Step 13: Print results.
    print(f"Expectation value:       {expec}")
    print(f"Final number of terms:   {in_expansion.num_terms}")
    print(f"Maximum number of terms: {max_num_terms}")
    print(f"Rehearsed requirements:")
    print(max_rehearsal_info)
    print(f"Runtime (s):             {duration}")
    print()


if __name__ == "__main__":
    main()

