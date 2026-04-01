#!/usr/bin/env python3
# Copyright (c) 2026-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Reverse-mode automatic differentiation of a kicked Ising expectation value.

An example demonstrating use of cuPauliProp's automatic differentiation facilities for
calculating expectation gradients of parameterised quantum circuits. We use the circuit
of IBM's 127-qubit kicked Ising experiment, as presented in Nature volume 618, pages 
500–505 (2023), without simulated noise or twirling. We consider observable Z_62 and the 
circuit of 20 Trotter steps, as correspond to Fig 4. b), and compute its expectation
gradient at the point where all X rotations have angle PI/4 and all Z rotations are -PI/2.

Precisely, we compute the two-element gradient [∂<Z>/∂θ1, ∂<Z>/∂θ2] of the function
<Z> = <0|U†(θ1,θ2) Z_62 U(θ1,θ2)|0> where θ1 and θ2 are the angles of all X and Z rotations
respectively, obtained at the point (θ1 = PI/4, θ2 = -PI/2). We compute this efficiently
using reverse-mode AD through the cuPauliProp backward differentiation API. The expectation
value is always evaluated in the Heisenberg picture, evolving the observable Z_62
through the adjoint circuit, as suggested by <Z> = Tr( U†(θ1,θ2) Z_62 U(θ1,θ2) |0><0| ).

Instead of caching every intermediate Pauli expansion during the AD forward pass, we
exploit the reversibility of Pauli rotations (via their adjoint) by replaying the
computation tape in reverse order. At each step we:

  1. Undo the forward gate (originally applied with adjoint=True) by applying its
     inverse (adjoint=False) to recover the intermediate expansion originally given
     as input to the forward step.
  2. Call apply_gate_backward_diff() on that recovered intermediate expansion to obtain
     the parameter gradient and the cotangent for the preceding layer.

This trades extra gate applications for O(1) memory in the number of circuit layers.
"""

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

NUM_CIRCUIT_QUBITS = 127
PI = np.pi

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
    print("cuPauliProp Kicked Ising — Reverse-Mode AD Example")
    print("=" * 55)
    print()

    handle = LibraryHandle()

    # Observable: Z_62 with coefficient 1.0
    num_packed = get_num_packed_integers(NUM_CIRCUIT_QUBITS)
    xz = cp.zeros((1, 2 * num_packed), dtype=cp.uint64)
    coefs = cp.zeros((1,), dtype=cp.float64)
    xz[0] = cp.asarray(get_pauli_string_as_packed_integers(["Z"], [62], NUM_CIRCUIT_QUBITS))
    coefs[0] = 1.0

    options = PauliExpansionOptions(memory_limit="80%", blocking=True)
    observable = PauliExpansion(handle, NUM_CIRCUIT_QUBITS, 1, xz, coefs, options=options)

    truncation = Truncation(pauli_coeff_cutoff=1e-4, pauli_weight_cutoff=8)

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

    # ================================================================
    # Forward pass: back-propagate observable through adjoint circuit.
    #
    # The forward computation is:
    #   E_0 = observable (Z_62)
    #   E_1 = E_0.apply_gate(circuit[N-1], adjoint=True)
    #   E_2 = E_1.apply_gate(circuit[N-2], adjoint=True)
    #   ...
    #   E_N = E_{N-1}.apply_gate(circuit[0], adjoint=True)
    #   loss = E_N.trace_with_zero_state()
    # ================================================================
    print("Forward pass ...")
    fwd_start = cp.cuda.Event()
    fwd_end = cp.cuda.Event()
    fwd_start.record()

    expansion = observable
    for gate_index in range(num_gates - 1, -1, -1):
        gate = circuit[gate_index]
        expansion = expansion.apply_gate(
            gate, truncation=truncation, adjoint=True,
            sort_order=None, keep_duplicates=False,
        )

    trace_significand, trace_exponent = expansion.trace_with_zero_state()
    expec = trace_significand * np.exp2(trace_exponent)

    fwd_end.record()
    fwd_end.synchronize()
    fwd_time = cp.cuda.get_elapsed_time(fwd_start, fwd_end) / 1000.0

    print(f"  Expectation value: {expec}")
    print(f"  Final num terms:   {expansion.num_terms}")
    print(f"  Forward time (s):  {fwd_time:.3f}")
    print()

    # ================================================================
    # Backward pass: reverse-mode AD.
    #
    # We walk the computation tape in reverse order.  The forward tape
    # applied gates as: circuit[N-1], circuit[N-2], ..., circuit[0]
    # (each with adjoint=True).  So the backward tape processes them
    # in order: circuit[0], circuit[1], ..., circuit[N-1].
    #
    # At each step k (processing circuit[k]):
    #   1. Undo the forward gate: since the forward step applied
    #      circuit[k] with adjoint=True, we apply circuit[k] with
    #      adjoint=False to recover the previous intermediate.
    #   2. Call apply_gate_backward_diff on the recovered intermediate
    #      (= viewIn of the forward step) with the current cotangent
    #      to get the parameter gradient and the cotangent for the
    #      preceding layer.
    # ================================================================
    print("Backward pass ...")
    bwd_start = cp.cuda.Event()
    bwd_start.record()

    # Seed: backward through trace_with_zero_state using chain rule for
    # trace = s * 2^p, where s=traceSignificand and p=traceExponent:
    #   dL/ds = g * 2^p
    #   dL/df = g * s * 2^p * ln(2)
    # For this example L(trace)=trace so upstream seed g = dL/dtrace = 1.
    upstream_seed = 1.0
    cotangent_trace_significand = upstream_seed * np.exp2(trace_exponent)
    cotangent_trace_exponent = (
        upstream_seed
        * trace_significand
        * np.exp2(trace_exponent)
        * np.log(2.0)
    )
    cotangent_exp = expansion.trace_with_zero_state_backward_diff(
        cotangent_trace_significand,
        cotangent_trace_exponent,
    )

    grad_x_angle = 0.0
    grad_zz_angle = 0.0

    for gate_index in range(num_gates):
        gate = circuit[gate_index]

        # Step 1: undo the forward gate to recover the previous intermediate.
        # Truncation must also be applied here to prevent unbounded growth;
        # the forward intermediates were truncated, so the reversal should be too.
        expansion = expansion.apply_gate(
            gate, truncation=truncation, adjoint=False,
            sort_order=None, keep_duplicates=False,
        )

        # Step 2: backward through this gate application.
        # expansion is now the viewIn; cotangent_exp is the cotangent of the output.
        new_cotangent_exp, param_grads = expansion.apply_gate_backward_diff(
            gate=gate,
            cotangent_out=cotangent_exp.view(),
            truncation=truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )

        cotangent_exp = new_cotangent_exp
        del new_cotangent_exp

        if param_grads is not None:
            if len(gate.pauli_string) == 1 and gate.pauli_string[0] == "X":
                grad_x_angle += param_grads[0]
            else:
                grad_zz_angle += param_grads[0]

    bwd_end = cp.cuda.Event()
    bwd_end.record()
    bwd_end.synchronize()
    bwd_time = cp.cuda.get_elapsed_time(bwd_start, bwd_end) / 1000.0

    print(f"  d<Z_62>/d(x_angle):  {grad_x_angle}")
    print(f"  d<Z_62>/d(zz_angle): {grad_zz_angle}")
    print(f"  Backward time (s):   {bwd_time:.3f}")
    print()


if __name__ == "__main__":
    main()
