# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import stim

from cuquantum.stabilizer.dem_sampling import DEMSampler
from cuquantum.stabilizer.simulator import Options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--prob", type=float, default=0.001)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bit-packed", action="store_true")
    ap.add_argument("--method", default=None, choices=["sparse", "dense"],
                    help="Override sampling method (default: auto)")
    ns = ap.parse_args()

    rounds = ns.distance if ns.rounds is None else ns.rounds
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=ns.distance,
        rounds=rounds,
        after_clifford_depolarization=ns.prob,
        before_round_data_depolarization=ns.prob,
        before_measure_flip_probability=ns.prob,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    ).flattened()

    sampler = DEMSampler(dem, ns.shots, options=Options(device_id=0))
    sampler.sample(ns.shots, seed=ns.seed)

    if ns.bit_packed:
        packed = sampler.get_outcomes(bit_packed=True)
        print(f"Bit-packed shape: {packed.shape}, dtype: {packed.dtype}")
    else:
        dense = sampler.get_outcomes(bit_packed=False)
        pops = dense.get().sum(axis=0) / float(ns.shots)
        print(f"Detector probabilities:\n{pops}")



if __name__ == "__main__":
    main()

