#!/usr/bin/env python3
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: IBM 127-qubit Kicked Ising simulation using the pythonic cuPauliProp API.

This ports the bindings sample in ``samples/bindings/cupauliprop/kicked_ising_example.py``
to the higher-level pythonic API. It back-propagates the Z_62 observable through the
127-qubit heavy-hex Ising circuit (20 Trotter steps, X angle pi/4) and evaluates the
expectation value with respect to initial state |0...0>.
"""
# Sphinx
from typing import Sequence

import cupy as cp
import numpy as np

from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)

# Circuit constants (IBM heavy-hex kicked Ising)
NUM_CIRCUIT_QUBITS = 127
NUM_ROTATIONS_PER_LAYER = 48
PI = np.pi
ZZ_ROTATION_ANGLE = -PI / 2.0

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
    print("cuPauliProp IBM Heavy-hex Ising Example (pythonic API, simple)")
    print("=" * 68)
    print()

    # Step 1: Create a library handle.
    handle = LibraryHandle()

    # Step 2: Create the initial observable (Z_62 with coefficient 1.0).
    num_packed = get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    xz = cp.zeros((1, 2 * num_packed), dtype=cp.uint64)
    coefs = cp.zeros((1,), dtype=cp.float64)
    xz[0] = cp.asarray(get_pauli_string_as_packed_integers(["Z"], [62], NUM_CIRCUIT_QUBITS))
    coefs[0] = 1.0

    # Step 3: Create a PauliExpansion with the observable.
    options = PauliExpansionOptions(memory_limit="80%", blocking=True)
    expansion = PauliExpansion(
        handle,
        NUM_CIRCUIT_QUBITS,
        1,
        xz,
        coefs,
        options=options,
    )

    # Step 4: Define truncation strategy to control term growth.
    truncation = Truncation(pauli_coeff_cutoff=1e-4, pauli_weight_cutoff=8)
    num_gates_between_truncations = 10

    # Step 5: Build the quantum circuit (kicked Ising model).
    x_rotation_angle = PI / 4.0
    num_trotter_steps = 20
    circuit = get_ibm_heavy_hex_ising_circuit(x_rotation_angle, num_trotter_steps)

    print(f"Circuit: 127-qubit IBM heavy-hex Ising")
    print(f"  Trotter steps: {num_trotter_steps}")
    print(f"  Total gates:   {len(circuit)}")
    print(f"  Rx angle:      {x_rotation_angle} (pi/4)")
    print()

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    max_num_terms = 0

    # Step 6: Back-propagate through the adjoint circuit.
    for gate_index in range(len(circuit) - 1, -1, -1):
        gate = circuit[gate_index]

        # Apply truncation periodically to control term growth.
        active_truncation = truncation if (gate_index % num_gates_between_truncations == 0) else None
        expansion = expansion.apply_gate(
            gate,
            truncation=active_truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )

        # Track maximum output term count after each gate.
        max_num_terms = max(max_num_terms, expansion.num_terms)

    # Step 7: Compute the expectation value <Z_62> with respect to |0...0>.
    trace_significand, trace_exponent = expansion.trace_with_zero_state()
    expec = trace_significand * np.exp2(trace_exponent)

    end_event.record()
    end_event.synchronize()
    duration = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0  # seconds

    # Step 8: Print results.
    print(f"Expectation value:       {expec}")
    print(f"Final number of terms:   {expansion.num_terms}")
    print(f"Maximum number of terms: {max_num_terms}")
    print(f"Runtime (s):             {duration}")
    print()


if __name__ == "__main__":
    main()

