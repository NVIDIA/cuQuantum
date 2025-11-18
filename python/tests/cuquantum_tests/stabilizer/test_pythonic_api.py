# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pythonic stabilizer API."""

import pytest
import numpy as np
import math
import stim
import time
from typing import Union
import logging
from enum import Enum

logger = logging.getLogger("pythonic")
log = logger.debug

Array = Union[np.ndarray, "cp.ndarray"]

try:
    import cupy as cp
except ImportError:
    cp = np
from cuquantum.stabilizer import Circuit, FrameSimulator, Options


def test_circuit_smoke():
    """Test creating a circuit."""
    circ = Circuit("H 0\nCNOT 0 1\nM 0 1")
    assert circ.circuit is not None


def test_frame_simulator_smoke():
    """Test creating a circuit."""
    sim = FrameSimulator(2, 1024, num_measurements=1)
    assert sim.num_qubits == 2
    sim = FrameSimulator(2, 1027, num_measurements=1)
    assert sim.num_paulis == 1027
    sim = FrameSimulator(0, 27, num_measurements=15)
    assert sim.num_measurements == 15


def test_simulation_basic():
    """Test creating a circuit."""
    circ = Circuit("X_ERROR(1) 0\nZ_ERROR(1) 1\nH 0 1\nCNOT 1 2\n M 2 3\n")
    possible = ("ZXY.", "ZXX.", "ZXYZ", "ZXXZ")
    sim = FrameSimulator(len(possible[0]), 1024, num_measurements=2)
    sim.apply(circ)
    table = sim.get_pauli_table()
    assert table[0].to_string() in possible
    assert table[487].to_string() in possible
    assert table[1023].to_string() in possible
    assert sim.num_qubits == len(possible[0])
    mbits: Array = sim.get_measurement_bits()
    assert (mbits[0] == 255).all()
    assert (mbits[1] == 0).all()
    mbits: Array = sim.get_measurement_bits(bit_packed=False)
    assert len(mbits[0]) == 1024
    assert (mbits[0] == 1).all()
    assert (mbits[1] == 0).all()


def calculate_table_population(*args):
    t = tuple(a.sum(axis=-1) for a in args)
    return np.concatenate(t)


def stim_circuit_filter_gates(circuit: stim.Circuit, gates: list[str]) -> stim.Circuit:
    new_circuit = stim.Circuit()
    for gate in circuit:
        if isinstance(gate, stim.CircuitRepeatBlock):
            newbloc_body = stim_circuit_filter_gates(gate.body_copy(), gates)
            newblock = stim.CircuitRepeatBlock(gate.repeat_count, newbloc_body, tag=gate.tag)
            new_circuit.append(newblock)
            continue
        if gate.name not in gates:
            new_circuit.append(gate)
            new_circuit.append('tick')

    return new_circuit


class Circuits(Enum):
    rare_events1 = """
       REPEAT 30 {
           X_ERROR(0.001) 0 1 2 5
           Z_ERROR(0.001) 0 4 1 3
           DEPOLARIZE1(0.005) 0 4
           DEPOLARIZE2(0.006) 3 7 6 0
       }
       M(0.009) 0 2 4
       MRY(0.009) 7 5
       """


def get_circuit(circuit_name: str, d, r, p) -> str:
    if "memory" in circuit_name:
        circuit = str(stim.Circuit.generated(
            "surface_code:" + circuit_name,
            distance=d,
            rounds=r,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
        ))
    else:
        circuit = str(Circuits[circuit_name].value)
    return circuit

@pytest.mark.parametrize(
    ("d", "r", "p", "nshots", "circuit_name", "randomize_measurements"),
    # fmt: off
    [
        ( 4, 4, 0.001, 1024 * 200, "rotated_memory_z",          False,),     #
        ( 7, 3, 0.01,  1024 * 80,  "rotated_memory_x",          False,),     #
        ( 8, 9, 0.002, 1024 * 100, "unrotated_memory_z",        False,),     #
        ( 5, 4, 0.1,   32 * 357,   "unrotated_memory_x",        True, ),     #
        ( 0, 0, 0,     32 * 357,   Circuits.rare_events1.name,  False,),     #
        ( 0, 0, 0,     128     ,   Circuits.rare_events1.name,  False,),     #
     ],
    # fmt: on
)
def test_statistical_wrt_stim(d, r, p, nshots, circuit_name, randomize_measurements):
    log(
        f"Surface code test {d=} {r=} {p=} {nshots=} {circuit_name=} {randomize_measurements=}"
    )
    circuit = stim.Circuit(get_circuit(circuit_name, d, r, p))
    cuda_circuit = Circuit(circuit)
    sim = FrameSimulator(
        circuit.num_qubits,
        nshots,
        circuit.num_measurements,
        num_detectors=circuit.num_detectors,
        randomize_measurements=randomize_measurements,
        seed=0,
        package="cupy",
    )
    sim.apply(cuda_circuit)
    xzbits = sim.get_pauli_xz_bits(bit_packed=False)
    mbits = sim.get_measurement_bits(bit_packed=False)

    def get_stim_probs(seed):
        sim_ref = stim.FlipSimulator(
            num_qubits=circuit.num_qubits,
            batch_size=nshots,
            seed=seed,
            disable_stabilizer_randomization=not randomize_measurements,
        )
        stim_start = time.time()
        sim_ref.do(circuit)
        stim_end = time.time()
        log(f"Stim time: {(stim_end - stim_start) * 1000} ms")
        xref, zref, mref, dref, oref = sim_ref.to_numpy(
            bit_packed=False, output_xs=True, output_zs=True, output_measure_flips=True
        )
        probs_ref = calculate_table_population(xref, zref, mref) / nshots
        return probs_ref

    prob = calculate_table_population(xzbits[0], xzbits[1], mbits) / nshots
    prob = prob.get()
    prob_ref = get_stim_probs(0)
    log(f"Shapes: {prob.shape=}, {prob_ref.shape=}")
    print_num = 30

    num_tries = 4
    retry_violations_below = 5
    K = len(prob)
    # TODO: this is still giving false negatives for small p
    z = np.sqrt(2 * np.log(K) + 15)
    log(f"Using statistical z={z} for {K} probabilities")
    individual_FP = math.erfc(z / np.sqrt(2))
    log(f"Individual false positive rate: {individual_FP}")
    log(f"Joint false positive rate: {1 - (1 - individual_FP) ** K}")

    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.5f" % x)
        )

        violations_by_try: dict[int, tuple] = {}
        for tryix in range(num_tries):
            log("IX=%s", ' '.join(f"{ix:7}" for ix in range(min(print_num, K))))
            log(f"prob={prob[:print_num]}")
            log(f"pref={prob_ref[:print_num]}")
            assert not np.all(prob == 0), "Probs should not be all zeros"
            assert np.all((prob_ref == 0) == (prob == 0))

            diff = np.abs(prob_ref - prob)
            log(f"diff={diff[:print_num]}")
            sigma = np.std(prob_ref[prob_ref != 0])
            atol = sigma * 2 / np.sqrt(nshots)

            pref = prob_ref
            atol = z * np.sqrt(pref * (1 - pref) / nshots)  # vector atol
            rtol = 1 / nshots
            log(f"atol={atol[:print_num]}")
            log(f"rtol={rtol}")
            max_diff = np.max(diff)
            max_diff_ix = np.argmax(diff)
            log(
                f"Max difference: {max_diff} at {max_diff_ix} ({prob_ref[max_diff_ix]}(ref) vs {prob[max_diff_ix]})"
            )

            close = np.isclose(prob, pref, atol=atol, rtol=rtol)
            violations = np.where(~close)[0]
            if len(violations) != 0:
                logger.warning(f"{len(violations)} violations at try {tryix}!")
                log(f"Violations: {violations[:print_num]}")
                log(f"Values res: {prob[violations][:print_num]}")
                log(f"Values ref: {pref[violations][:print_num]}")
                log(f"diff      : {diff[violations][:print_num]}")
                log(f"atol      : {atol[violations][:print_num]}")
                assert len(violations) < retry_violations_below
                violations_by_try[tryix] = tuple(violations.tolist())
                prob_ref = get_stim_probs(tryix + 1)
            else:
                break

        assert len(violations_by_try) < num_tries, "All retries had violations"
    finally:
        np.set_printoptions(**original_printoptions)


def test_multiple_circuits_same_simulator():
    """Test reusing same simulator for multiple circuits."""
    circ = Circuit(
        """
       X_ERROR(0.1) 0 2 5
       Z_ERROR(0.3) 1 2 4
       H 0 1 3
       CNOT 0 1 5 2
       X_ERROR(0.002) 0 1 2 5
       Z_ERROR(0.002) 0 1 2 5
       DEPOLARIZE2(0.005) 1 4
       M 0 2 4
       """
    )
    nshots = 1024 * 5
    nqubits = 6
    nmeas = 3
    sim = FrameSimulator(nqubits, nshots, num_measurements=nmeas)

    seed = 15
    # Apply first circuit
    sim.apply(circ, seed=seed)
    m1 = sim.get_measurement_bits(bit_packed=False)
    x1, z1 = sim.get_pauli_xz_bits()
    x1, z1 = x1.copy(), z1.copy()

    sim.apply(circ, seed=seed)
    m2 = sim.get_measurement_bits(bit_packed=False)
    x2, z2 = sim.get_pauli_xz_bits()
    x2, z2 = x2.copy(), z2.copy()

    # Reset tables and apply second circuit
    x_table = np.zeros((nqubits, nshots), dtype=np.uint8)
    z_table = np.zeros((nqubits, nshots), dtype=np.uint8)
    m_table = np.zeros((nmeas, nshots // 8), dtype=np.uint8)  # bit-packed format

    sim.set_input_tables(x_table, z_table, bit_packed=False)
    sim.set_input_tables(x=None, z=None, m=m_table, bit_packed=True)
    sim.apply(circ, seed=seed)
    m3 = sim.get_measurement_bits(bit_packed=False)
    x3, z3 = sim.get_pauli_xz_bits()
    assert x1.shape == x2.shape

    assert not np.array_equal(x1, x2)
    assert not np.array_equal(z1, z2)
    assert not np.array_equal(m1, m2)
    assert not np.all(m1 == 0)
    assert not np.all(m2 == 0)
    assert not np.all(m3 == 0)
    assert np.array_equal(x1, x3)
    assert np.array_equal(z1, z3)
    assert np.array_equal(m1, m3)

def test_multiple_runs_same_simulator():
    """
    Concatenate results of multiple runs and compare against one run with nshots=sum(nshots_small).

    Notes:
    - The circuit should contain small probabilities to trigger rare event sampling
    - Number of small runs should be large enough to avoid false failures of the test
    - Since the gate errors are small, use REPEAT instruction to accumulate errors
    """
    circ = Circuit(Circuits.rare_events1.value)
    stim_circ = stim.Circuit(circ.circuit_string)

    small_runs = [1] * 128
    big_run = sum(small_runs)
    nqubits = stim_circ.num_qubits
    nmeas = stim_circ.num_measurements
    logger = logging.getLogger("Ignore")
    logger.setLevel(level=logging.WARNING)
    options = Options(logger=logger)
    seed = 10
    sim_small = FrameSimulator(
        nqubits,
        max(small_runs),
        num_measurements=nmeas,
        randomize_measurements=False,
        seed=seed,
        options=options,
    )
    sim_big = FrameSimulator(
        nqubits,
        big_run,
        num_measurements=nmeas,
        randomize_measurements=False,
        seed=seed,
    )

    def reset(sim, nqubits, nmeas, nshots):
        x_table = np.zeros((nqubits, nshots), dtype=np.uint8)
        z_table = np.zeros((nqubits, nshots), dtype=np.uint8)
        m_table = np.zeros((nmeas, nshots), dtype=np.uint8)
        sim.set_input_tables(x=x_table, z=z_table, m=m_table, bit_packed=False)

    sm_results = []
    for i, run in enumerate(small_runs):
        reset(sim_small, nqubits, nmeas, max(small_runs))
        sim_small.apply(circ)
        x, z = sim_small.get_pauli_xz_bits(bit_packed=False)
        mbits = sim_small.get_measurement_bits(bit_packed=False)
        # log(f"run={i} xbits={x.flatten()} zbits={z.flatten()} mbits={mbits.flatten()}")
        counts_i = calculate_table_population(x[:, :run], z[:, :run], mbits[:, :run])
        # log(f"run={i} counts={counts_i}")
        sm_results.append(counts_i)

    sim_big.apply(circ)
    xzbits = sim_big.get_pauli_xz_bits(bit_packed=False)
    mbits = sim_big.get_measurement_bits(bit_packed=False)
    probs_big = calculate_table_population(xzbits[0], xzbits[1], mbits) / big_run
    probs_sm = np.sum(sm_results, axis=0) / np.sum(small_runs)
    print_num = 20
    original_printoptions = np.get_printoptions()
    try:
        np.set_printoptions(
            edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.6f" % x)
        )
        log(f"probs_big={probs_big[:print_num]}")
        log(f"probs_sml={probs_sm[:print_num]}")

        K = len(probs_big)
        z = np.sqrt(2 * np.log(K) + 15)
        atol = z * np.sqrt(probs_sm * (1 - probs_sm) / big_run)  # vector atol
        rtol = 1 / big_run
        diff = np.abs(probs_big - probs_sm)
        log(f"diff = {diff[:print_num]}")
        log(f"atol = {atol[:print_num]} rtol={rtol}")
        assert np.allclose(probs_big, probs_sm, atol=atol, rtol=rtol)
    finally:
        np.set_printoptions(**original_printoptions)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
