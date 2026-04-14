# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import statistics
import time

from cuquantum.tensornet import CircuitToEinsum


def build_qiskit_circuit(num_qubits, depth):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_qubits)
    for layer in range(depth):
        for qubit in range(num_qubits):
            angle = 0.1 * (layer + 1) * (qubit + 1)
            qc.ry(angle, qubit)
            qc.rz(angle / 2, qubit)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    return qc


def build_cirq_circuit(num_qubits, depth):
    import cirq

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    for layer in range(depth):
        for qubit in qubits:
            angle = 0.1 * (layer + 1) * (qubit.x + 1)
            circuit.append(cirq.ry(angle)(qubit))
            circuit.append(cirq.rz(angle / 2)(qubit))
        for left, right in zip(qubits, qubits[1:]):
            circuit.append(cirq.CNOT(left, right))
    return circuit


def build_circuit(framework, num_qubits, depth):
    if framework == "qiskit":
        return build_qiskit_circuit(num_qubits, depth)
    if framework == "cirq":
        return build_cirq_circuit(num_qubits, depth)
    raise ValueError(f"unsupported framework: {framework}")


def benchmark_expectation(converter, repetitions, lightcone):
    qubits = converter.qubits
    active_qubits = qubits[: min(3, len(qubits))]
    pauli_map = {qubit: ("Z" if i % 2 == 0 else "X") for i, qubit in enumerate(active_qubits)}

    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        converter.expectation(pauli_map, lightcone=lightcone)
        timings.append(time.perf_counter() - start)
    return timings


def benchmark_rdm(converter, repetitions, lightcone):
    qubits = converter.qubits[: min(3, len(converter.qubits))]
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        converter.reduced_density_matrix(qubits, lightcone=lightcone)
        timings.append(time.perf_counter() - start)
    return timings


def summarize_timings(name, timings):
    cold = timings[0]
    warm = timings[1:] if len(timings) > 1 else timings
    warm_mean = statistics.mean(warm)
    speedup = cold / warm_mean if warm_mean else float("inf")

    print(f"{name}:")
    print(f"  cold call:  {cold:.6f} s")
    print(f"  warm mean:  {warm_mean:.6f} s")
    print(f"  warm min:   {min(warm):.6f} s")
    print(f"  warm max:   {max(warm):.6f} s")
    print(f"  speedup:    {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CircuitToEinsum metadata cache reuse.")
    parser.add_argument("--framework", choices=("qiskit", "cirq"), default="qiskit")
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--backend", default="numpy")
    args = parser.parse_args()

    circuit = build_circuit(args.framework, args.qubits, args.depth)
    converter = CircuitToEinsum(circuit, dtype=args.dtype, backend=args.backend)

    print("CircuitToEinsum metadata cache benchmark")
    print(f"framework:   {args.framework}")
    print(f"qubits:      {args.qubits}")
    print(f"depth:       {args.depth}")
    print(f"repetitions: {args.repetitions}")
    print(f"dtype:       {args.dtype}")
    print(f"backend:     {args.backend}")
    print()

    expectation_lightcone = benchmark_expectation(converter, args.repetitions, lightcone=True)
    expectation_no_lightcone = benchmark_expectation(converter, args.repetitions, lightcone=False)
    rdm_lightcone = benchmark_rdm(converter, args.repetitions, lightcone=True)
    rdm_no_lightcone = benchmark_rdm(converter, args.repetitions, lightcone=False)

    summarize_timings("expectation(lightcone=True)", expectation_lightcone)
    summarize_timings("expectation(lightcone=False)", expectation_no_lightcone)
    summarize_timings("reduced_density_matrix(lightcone=True)", rdm_lightcone)
    summarize_timings("reduced_density_matrix(lightcone=False)", rdm_no_lightcone)


if __name__ == "__main__":
    main()
