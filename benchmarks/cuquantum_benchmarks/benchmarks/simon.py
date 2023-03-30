# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

# Used for classical post-processing:
from collections import Counter

import numpy as np
import scipy as sp

from .benchmark import Benchmark
from .._utils import Gate, reseed


class Simon(Benchmark):

    # Example instantiation of Simon circuit paramterized by nqubits
    @staticmethod
    def generateGatesSequence(nqubits, config):
        reseed(1234)  # TODO: use a global seed?

        # "nqubits" here refers to the number of total qubits in the circuit, and we need
        # it to be even so we can split the qubits into input/output
        if nqubits % 2:
            raise ValueError("the simon benchmark requires even number of qubits")
        elif nqubits < 4:
            # because the oracle needs to apply a swap gate, # output qubits needs to be
            # at least 2, so total # of qubits needs to be 4
            raise ValueError("the simon benchmark requires at least 4 qubits")
        else:
            nqubits //= 2

        measure = config['measure']
        # define a secret string:
        secret_string = np.random.randint(2, size=nqubits)
        # Choose qubits to use.
        input_qubits = [i for i in range(nqubits)]  # input x
        output_qubits = [j for j in range(nqubits, 2*nqubits)] # output f(x)
        # Pick coefficients for the oracle and create a circuit to query it.
        oracle = Simon._make_oracle(input_qubits, output_qubits, secret_string)
        # Embed oracle into special quantum circuit querying it exactly once
        circuit = Simon._make_simon_circuit(input_qubits, output_qubits, oracle, measure)
        return circuit

    """Demonstrates Simon's algorithm.
    Simon's Algorithm solves the following problem:

    Given a function  f:{0,1}^n -> {0,1}^n, such that for some s ∈ {0,1}^n,

    f(x) = f(y) iff  x ⨁ y ∈ {0^n, s},

    find the n-bit string s.

    A classical algorithm requires O(2^n/2) queries to find s, while Simon’s
    algorithm needs only O(n) quantum queries.

    === REFERENCE ===
    D. R. Simon. On the power of quantum cryptography. In35th FOCS, pages 116–123,
    Santa Fe,New Mexico, 1994. IEEE Computer Society Press.

    === EXAMPLE OUTPUT ===
    Secret string = [1, 0, 0, 1, 0, 0]
    Circuit:
                    ┌──────┐   ┌───────────┐
    (0, 0): ────H────@──────────@─────@──────H───M('result')───
                     │          │     │          │
    (1, 0): ────H────┼@─────────┼─────┼──────H───M─────────────
                     ││         │     │          │
    (2, 0): ────H────┼┼@────────┼─────┼──────H───M─────────────
                     │││        │     │          │
    (3, 0): ────H────┼┼┼@───────┼─────┼──────H───M─────────────
                     ││││       │     │          │
    (4, 0): ────H────┼┼┼┼@──────┼─────┼──────H───M─────────────
                     │││││      │     │          │
    (5, 0): ────H────┼┼┼┼┼@─────┼─────┼──────H───M─────────────
                     ││││││     │     │
    (6, 0): ─────────X┼┼┼┼┼─────X─────┼───×────────────────────
                      │││││           │   │
    (7, 0): ──────────X┼┼┼┼───────────┼───┼────────────────────
                       ││││           │   │
    (8, 0): ───────────X┼┼┼───────────┼───┼────────────────────
                        │││           │   │
    (9, 0): ────────────X┼┼───────────X───×────────────────────
                         ││
    (10, 0): ────────────X┼────────────────────────────────────
                          │
    (11, 0): ─────────────X────────────────────────────────────
                    └──────┘   └───────────┘
    Most common Simon Algorithm answer is: ('[1 0 0 1 0 0]', 100)

    ***If the input string is s=0^n, no significant answer can be
    distinguished (since the null-space of the system of equations
    provided by the measurements gives a random vector). This will
    lead to low frequency count in output string.
    """

    def _make_oracle(input_qubits, output_qubits, secret_string):
        """Gates implementing the function f(a) = f(b) iff a ⨁ b = s"""
        # Copy contents to output qubits:
        for control_qubit, target_qubit in zip(input_qubits, output_qubits):
            yield Gate(id='cnot', controls=control_qubit, targets=target_qubit)

        # Create mapping:
        if sum(secret_string):  # check if the secret string is non-zero
            # Find significant bit of secret string (first non-zero bit)
            significant = list(secret_string).index(1)
            # Add secret string to input according to the significant bit:
            for j in range(len(secret_string)):
                if secret_string[j] > 0:
                    yield Gate(id='cnot', controls=input_qubits[significant], targets=output_qubits[j])

        # Apply a random permutation:
        pos = [0, len(secret_string) - 1,]
        # Swap some qubits to define oracle. We choose first and last:
        yield Gate(id='swap', targets=[output_qubits[pos[0]], output_qubits[pos[1]]])

    def _make_simon_circuit(input_qubits, output_qubits, oracle, measure):
        """Solves for the secret period s of a 2-to-1 function such that
        f(x) = f(y) iff x ⨁ y = s
        """
        circuit = []
        # Initialize qubits.
        init = [Gate(id='h', targets=idx) for idx in input_qubits]
        circuit += init

        # Query oracle.
        circuit += oracle

        if measure:
            # Measure in X basis.
            circuit += init
            circuit.append(Gate(id='measure', targets=input_qubits))

        return circuit

    def _post_processing(data, results):
        """Solves a system of equations with modulo 2 numbers"""
        sing_values = sp.linalg.svdvals(results)
        tolerance = 1e-5
        if sum(sing_values < tolerance) == 0:  # check if measurements are linearly dependent
            flag = True
            null_space = sp.linalg.null_space(results).T[0]
            solution = np.around(null_space, 3)  # chop very small values
            minval = abs(min(solution[np.nonzero(solution)], key=abs))
            solution = (solution / minval % 2).astype(int)  # renormalize vector mod 2
            data.append(str(solution))
            return flag

    @staticmethod
    def postProcess(nqubits, results):
        if results is None:
            return False

        # only the input qubits are measured
        nqubits //= 2

        data = []
        flag = False
        classical_iter = 0
        while not flag:
            # Classical Post-Processing:
            flag = Simon._post_processing(data, results[classical_iter * (nqubits - 1):(classical_iter + 1) * (nqubits - 1)])
            classical_iter += 1
        freqs = Counter(data)
        return True
