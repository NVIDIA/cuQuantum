# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import stim
from cuquantum.stabilizer import Circuit, FrameSimulator, Options
import time


def main():
    parser = argparse.ArgumentParser(description="Surface code simulation example")
    parser.add_argument("--distance", type=int, default=9, help="Surface code distance (default: 9)")
    parser.add_argument("--rounds", type=int, default=9, help="Number of rounds (default: 9)")
    parser.add_argument("--prob", type=float, default=0.002, help="After Clifford depolarization probability (default: 0.002)")
    parser.add_argument("--shots", type=int, default=1024*1000, help="Number of shots (default: 1024000)")
    args = parser.parse_args()

    d = args.distance
    r = args.rounds
    p = args.prob
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=r,
        after_clifford_depolarization=p,
    )

    cuda_circuit = Circuit(circuit)
    nshots = args.shots

    sim = FrameSimulator(
        circuit.num_qubits,
        nshots,
        circuit.num_measurements,
        num_detectors=circuit.num_detectors,
        randomize_measurements=False,
    )
    start = time.time()
    sim.apply(cuda_circuit)
    table = sim.get_pauli_table()
    print("Simulation done")

    print_count = 5
    print("Custabilizer strings:")
    for i in range(print_count):
        print(table[i])
    end = time.time()
    print(f"Time: {end - start}")

    stsim = stim.FlipSimulator(batch_size=nshots, disable_stabilizer_randomization=True)
    start = time.time()
    stsim.do(circuit)
    t = stsim.peek_pauli_flips()
    print("Stim strings:")
    for i in range(print_count):
        print(t[i])
    end = time.time()
    print(f"Time: {end - start}")


if __name__ == "__main__":
    main()
