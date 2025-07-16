# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import random

from .benchmark import Benchmark
from .._utils import Gate, reseed


class HiddenShift(Benchmark):

    @staticmethod
    def generateGatesSequence(nqubits, config):
        reseed(1234)  # TODO: use a global seed?

        # Define secret shift
        shift = [random.randint(0, 1) for _ in range(nqubits)]

        # Make oracles (black box)
        oracle_f = HiddenShift._make_oracle_f(nqubits)

        # Embed oracle into quantum circuit implementing the Hidden Shift Algorithm
        circuit = HiddenShift._make_hs_circuit(nqubits, oracle_f, shift)

        measure = config['measure']
        if measure:
            circuit.append(Gate(id='measure', targets=list(range(nqubits))))

        return circuit

    """Example program that demonstrates a Hidden Shift algorithm.
    The Hidden Shift Problem is one of the known problems whose quantum algorithm
    solution shows exponential speedup over classical computing. Part of the
    advantage lies on the ability to perform Fourier transforms efficiently. This
    can be used to extract correlations between certain functions, as we will
    demonstrate here:

    Let f and g be two functions {0,1}^N -> {0,1}  which are the same
    up to a hidden bit string s:

    g(x) = f(x ⨁ s), for all x in {0,1}^N

    The implementation in this example considers the following (so-called "bent")
    functions:

    f(x) = Σ_i x_(2i) x_(2i+1),

    where x_i is the i-th bit of x and i runs from 0 to N/2 - 1.

    While a classical algorithm requires 2^(N/2) queries, the Hidden Shift
    Algorithm solves the problem in O(N) quantum operations. We describe below the
    steps of the algorithm:

    (1) Prepare the quantum state in the initial state |0⟩^N

    (2) Make a superposition of all inputs |x⟩ with  a set of Hadamard gates, which
    act as a (Quantum) Fourier Transform.

    (3) Compute the shifted function g(x) = f(x ⨁ s) into the phase with a proper
    set of gates. This is done first by shifting the state |x⟩ with X gates, then
    implementing the bent function as a series of Controlled-Z gates, and finally
    recovering the |x⟩ states with another set of X gates.

    (4) Apply a Fourier Transform to generate another superposition of states with
    an extra phase that is added to f(x ⨁ s).

    (5) Query the oracle f into the phase with a proper set of controlled gates.
    One can then prove that the phases simplify giving just a superposition with
    a phase depending directly on the shift.

    (6) Apply another set of Hadamard gates which act now as an Inverse Fourier
    Transform to get the state |s⟩

    (7) Measure the resulting state to get s.

    Note that we only query g and f once to solve the problem.

    === REFERENCES ===
    [1] Wim van Dam, Sean Hallgreen, Lawrence Ip Quantum Algorithms for some
    Hidden Shift Problems. https://arxiv.org/abs/quant-ph/0211140
    [2] K Wrigth, et. a. Benchmarking an 11-qubit quantum computer.
    Nature Communications, 107(28):12446–12450, 2010. doi:10.1038/s41467-019-13534-2
    [3] Rötteler, M. Quantum Algorithms for highly non-linear Boolean functions.
    Proceedings of the 21st annual ACM-SIAM Symposium on Discrete Algorithms.
    doi: 10.1137/1.9781611973075.37


    === EXAMPLE OUTPUT ===
    Secret shift sequence: [1, 0, 0, 1, 0, 1]
    Circuit:
    (0, 0): ───H───X───@───X───H───@───H───M('result')───
                       │           │       │
    (1, 0): ───H───────@───────H───@───H───M─────────────
                                           │
    (2, 0): ───H───────@───────H───@───H───M─────────────
                       │           │       │
    (3, 0): ───H───X───@───X───H───@───H───M─────────────
                                           │
    (4, 0): ───H───────@───────H───@───H───M─────────────
                       │           │       │
    (5, 0): ───H───X───@───X───H───@───H───M─────────────
    Sampled results:
    Counter({'100101': 100})
    Most common bitstring: 100101
    Found a match: True
    """
    def _make_oracle_f(nqubits):
        """Implement function {f(x) = Σ_i x_(2i) x_(2i+1)}."""
        oracle_circuit = [Gate(id='cz', controls=2*i, targets=2*i+1) for i in range(nqubits//2)]
        return oracle_circuit

    def _make_hs_circuit(nqubits, oracle_f, shift):
        """Find the shift between two almost equivalent functions."""
        circuit = []
        apply_h = [Gate(id='h', targets=idx) for idx in range(nqubits)]
        apply_shift = [Gate(id='x', targets=k) for k in range(len(shift)) if shift[k]]

        # Initialize qubits.
        circuit += apply_h

        # Query oracle g: It is equivalent to that of f, shifted before and after:
        # Apply Shift:
        circuit += apply_shift

        # Query oracle.
        circuit += oracle_f

        # Apply Shift:
        circuit += apply_shift

        # Second Application of Hadamards.
        circuit += apply_h

        # Query oracle f (this simplifies the phase).
        circuit += oracle_f

        # Inverse Fourier Transform with Hadamards to go back to the shift state:
        circuit += apply_h

        return circuit
