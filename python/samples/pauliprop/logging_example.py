# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal example demonstrating logging in the cuPauliProp pythonic API.

Run with:
    python logging_example.py
"""
# Sphinx
import logging
import cupy as cp

# Step 1: Configure logging BEFORE importing pauliprop.
# Use DEBUG for verbose output, INFO for milestone-only logging.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)-8s %(message)s'
)

from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliRotationGate,
    get_num_packed_integers,
)


def main():
    print("=" * 60)
    print("cuPauliProp Logging Example")
    print("=" * 60)
    print()

    # Step 2: Create a library handle (logs INFO on creation).
    handle = LibraryHandle()

    # Step 3: Prepare data for a simple Pauli expansion.
    num_qubits = 4
    num_terms = 1
    ints_per_string = get_num_packed_integers(num_qubits)

    xz_bits = cp.zeros((num_terms, 2 * ints_per_string), dtype=cp.uint64)
    coeffs = cp.ones(num_terms, dtype=cp.complex128)

    # Step 4: Create a PauliExpansion (logs INFO on creation).
    expansion = PauliExpansion(handle, num_qubits, num_terms, xz_bits, coeffs)

    # Step 5: Apply a gate (logs INFO on start/completion, DEBUG for details).
    gate = PauliRotationGate(angle=0.5, pauli_string="X", qubit_indices=[0])
    result = expansion.apply_gate(gate, sort_order=None)

    # Step 6: Compute trace (logs INFO on start/completion).
    trace_significand, trace_exponent = result.trace_with_zero_state()
    trace = trace_significand * pow(2.0, trace_exponent)
    print(f"\nTrace result: {trace}")

    # Step 7: Clean up (destructor logs DEBUG).
    print("\nCleaning up...")

    result = None
    expansion = None
    gate = None
    handle = None

    print("\nDone!")


if __name__ == "__main__":
    main()

