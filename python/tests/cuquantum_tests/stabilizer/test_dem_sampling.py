# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import pytest
import stim

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


from cuquantum.stabilizer import (
    DEMSampler,
    BitMatrixSampler,
    BitMatrixSparseSampler,
    BitMatrixCSR,
    Options,
)

pytestmark = pytest.mark.custabilizer


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_dem_sampler_shapes_packed_and_unpacked():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
        before_round_data_depolarization=0.001,
        before_measure_flip_probability=0.001,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    ).flattened()

    shots = 64
    sampler = DEMSampler(dem, shots, options=Options(device_id=0))
    sampler.sample(shots, seed=0)

    packed = sampler.get_outcomes(bit_packed=True)
    assert isinstance(packed, cp.ndarray)
    n_det = int(dem.num_detectors)
    n_det_padded = ((n_det + 31) // 32) * 32
    assert packed.shape == (shots, n_det_padded // 8)
    assert packed.dtype == cp.uint8

    dense = sampler.get_outcomes(bit_packed=False)
    assert isinstance(dense, cp.ndarray)
    assert dense.shape == (shots, n_det)
    assert dense.dtype == cp.uint8


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_matrix_sampler_numpy_io():
    n_errors = 5
    n_results = 7
    shots = 32

    check = np.zeros((n_errors, n_results), dtype=np.uint8)
    check[0, 0] = 1
    check[1, 1] = 1
    check[2, 2] = 1
    check[3, 3] = 1
    check[4, 4] = 1

    probs = np.asarray([0.0, 1.0, 0.5, 0.25, 0.75], dtype=np.float64)

    sampler = BitMatrixSparseSampler(check, probs, shots, options=Options(device_id=0))
    sampler.sample(shots, seed=0)

    packed = sampler.get_outcomes(bit_packed=True)
    assert isinstance(packed, np.ndarray)
    n_res_padded = ((n_results + 31) // 32) * 32
    assert packed.shape == (shots, n_res_padded // 8)
    assert packed.dtype == np.uint8

    dense = sampler.get_outcomes(bit_packed=False)
    assert isinstance(dense, np.ndarray)
    assert dense.shape == (shots, n_results)
    assert dense.dtype == np.uint8


def erfinv_winitzki(x, a=0.147):
    assert 0 <= x <= 1
    ln = np.log(1 - x**2)
    t = 2/(a * np.pi) + ln / 2
    return np.sqrt(np.sqrt(t**2 - ln/a) - t)

def z_from_confidence(K, confidence):
    p = 1 - confidence
    z = np.sqrt(2) * erfinv_winitzki(1 - p / K)
    return z

def acceptance_prob_normal_approx(N, y, q, *, confidence=0.995):
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    K = len(y)

    z = z_from_confidence(K, confidence)
    rtol = 1.0 / N
    V = y * (1.0 - y)
    atol = z * np.sqrt(np.maximum(V, 0.0) / N)
    tol = atol + rtol * np.abs(y)

    sigma = np.sqrt(np.maximum(q * (1.0 - q), 0.0) / N)

    def Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    per = np.zeros(K, dtype=float)
    for i in range(K):
        if sigma[i] == 0.0:
            per[i] = 1.0 if abs(q[i] - y[i]) <= tol[i] else 0.0
        else:
            a = (y[i] - tol[i] - q[i]) / sigma[i]
            b = (y[i] + tol[i] - q[i]) / sigma[i]
            per[i] = max(0.0, min(1.0, Phi(b) - Phi(a)))

    overall_accept = float(np.prod(per))
    return {"z": float(z), "per_coordinate_accept": per, "overall_accept": overall_accept}

def bernoulli_close(x, y, N, *, confidence=0.995, max_violations=0):
    K = len(x)
    z = z_from_confidence(K, confidence)
    single_fp = math.erfc(z / np.sqrt(2))
    print(f"{single_fp=}")
    print(f"{z=:.2f}")
    V = y * (1 - y)
    atol = z * np.sqrt(V / N)
    rtol = 1 / N
    close = np.isclose(x, y, atol=atol, rtol=rtol)
    violations = np.where(~close)[0]
    print_num = 10
    if len(violations) > 0:
        print(f"{len(violations)} violations: {violations[:print_num]}")
        print(f"x         : {x[violations][:print_num]}")
        print(f"y         : {y[violations][:print_num]}")
        print(f"atol      : {atol[violations][:print_num]}")
    return len(violations) <= int(max_violations)


def test_extract_from_dem_known():
    """Verify _extract_from_dem against a hand-constructed DEM covering:
    multi-detector errors, separators (^), and logical observable targets (L0).
    """
    from cuquantum.stabilizer.dem_sampling import _extract_from_dem

    dem = stim.DetectorErrorModel("""
        error(0.1) D0 D2
        error(0.25) D1
        error(0.5) D0 D1 ^ D2
        error(0.75) D2 L0
    """)

    n_det, probs, csr = _extract_from_dem(dem)

    assert n_det == 3
    assert csr.shape == (4, 3)
    assert csr.nnz == 7
    np.testing.assert_array_equal(probs, [0.1, 0.25, 0.5, 0.75])
    np.testing.assert_array_equal(csr.row_offsets, [0, 2, 3, 6, 7])
    np.testing.assert_array_equal(csr.col_indices, [0, 2, 1, 0, 1, 2, 2])


def test_correctness_small():
    """
    Take a square identity matrix (diagonal with ones) and use different probabilities for each.
    Verify that average population of sampled bit is equal to the probability.
    """
    n_errors = 65
    n_results = n_errors
    shots = 10000

    check = np.eye(n_errors, dtype=np.uint8)
    probs = np.random.rand(n_errors)

    sampler = BitMatrixSparseSampler(check, probs, shots, options=Options(device_id=0))
    sampler.sample(shots, seed=0)
    result = sampler.get_outcomes(bit_packed=False)

    pops = np.sum(result, axis=0) / shots
    assert bernoulli_close(pops, probs, shots, confidence=0.995)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("prob", [0.001, 0.005, 0.01])
def test_correctness_wrt_stim_dem_surface_code(distance, prob):
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=prob,
        before_round_data_depolarization=prob,
        before_measure_flip_probability=prob,
    )
    dem = circuit.detector_error_model(
        decompose_errors=True,
        approximate_disjoint_errors=True,
    ).flattened()

    shots = 50000
    seed = 7

    sampler = DEMSampler(dem, shots, options=Options(device_id=0), seed=seed)
    sampler.sample(shots, seed=seed)
    dets_cu = sampler.get_outcomes(bit_packed=False)
    pops_cu = cp.asnumpy(cp.sum(dets_cu, axis=0)) / float(shots)

    stim_sampler = dem.compile_sampler(seed=seed)
    dets_ref, _, _ = stim_sampler.sample(shots, bit_packed=False)
    pops_ref = dets_ref.mean(axis=0, dtype=np.float64)

    assert bernoulli_close(pops_cu, pops_ref, shots // 2, confidence=0.995, max_violations=2)


# --- Tests for get_errors and BitMatrixCSR ---

@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_csr_get_errors_returns_bit_matrix_csr():
    n_errors = 10
    n_results = 8
    shots = 64

    check = np.zeros((n_errors, n_results), dtype=np.uint8)
    for i in range(min(n_errors, n_results)):
        check[i, i] = 1
    probs = np.full(n_errors, 0.3, dtype=np.float64)

    sampler = BitMatrixSparseSampler(check, probs, shots, seed=42, package="cupy")
    sampler.sample(shots, seed=0)
    csr = sampler.get_errors()

    assert isinstance(csr, BitMatrixCSR)
    assert csr.shape == (shots, n_errors)
    assert csr.nnz >= 0
    assert csr.row_offsets.shape == (shots + 1,)
    assert csr.col_indices.shape[0] == csr.nnz
    assert csr.row_offsets.dtype == cp.int64
    assert csr.col_indices.dtype == cp.int64


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_csr_get_errors_numpy_package():
    n_errors = 10
    n_results = 8
    shots = 32

    check = np.zeros((n_errors, n_results), dtype=np.uint8)
    for i in range(min(n_errors, n_results)):
        check[i, i] = 1
    probs = np.full(n_errors, 0.3, dtype=np.float64)

    sampler = BitMatrixSparseSampler(check, probs, shots, seed=42, package="numpy")
    sampler.sample(shots, seed=0)
    csr = sampler.get_errors()

    assert isinstance(csr.row_offsets, np.ndarray)
    assert isinstance(csr.col_indices, np.ndarray)
    assert csr.row_offsets.dtype == np.int64
    assert csr.col_indices.dtype == np.int64


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_csr_to_scipy_sparse():
    n_errors = 10
    n_results = 8
    shots = 32

    check = np.zeros((n_errors, n_results), dtype=np.uint8)
    for i in range(min(n_errors, n_results)):
        check[i, i] = 1
    probs = np.full(n_errors, 0.5, dtype=np.float64)

    sampler = BitMatrixSparseSampler(check, probs, shots, seed=42, package="cupy")
    sampler.sample(shots, seed=0)
    csr = sampler.get_errors()

    scipy_sp = csr.to_scipy_sparse()
    assert scipy_sp.shape == (shots, n_errors)
    assert scipy_sp.nnz == csr.nnz


def _make_scipy_csr(n_rows, n_cols, rows, cols):
    import scipy.sparse
    data = np.ones(len(rows), dtype=np.uint8)
    return scipy.sparse.csr_array((data, (rows, cols)), shape=(n_rows, n_cols))


def _make_cupyx_csr(n_rows, n_cols, rows, cols):
    import cupy as cp
    import cupyx.scipy.sparse
    data = cp.ones(len(rows), dtype=cp.float32)
    r = cp.array(rows, dtype=cp.int32)
    c = cp.array(cols, dtype=cp.int32)
    return cupyx.scipy.sparse.csr_matrix((data, (r, c)), shape=(n_rows, n_cols))


@pytest.fixture
def sparse_test_data():
    n_errors, n_det = 6, 8
    rows = np.array([0, 0, 1, 2, 3, 4, 5])
    cols = np.array([0, 1, 2, 3, 4, 5, 6])
    probs = np.full(n_errors, 0.5, dtype=np.float64)
    return n_errors, n_det, rows, cols, probs


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
@pytest.mark.parametrize("make_csr", [_make_scipy_csr, _make_cupyx_csr], ids=["scipy", "cupyx"])
def test_csr_from_sparse(make_csr, sparse_test_data):
    n_errors, n_det, rows, cols, _ = sparse_test_data
    sp = make_csr(n_errors, n_det, rows, cols)

    csr = BitMatrixCSR.from_sparse(sp)
    assert csr.shape == (n_errors, n_det)
    assert csr.nnz == len(rows)

    rt = csr.to_scipy_sparse()
    assert rt.shape == (n_errors, n_det)
    assert rt.nnz == len(rows)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
@pytest.mark.parametrize("make_csr", [_make_scipy_csr, _make_cupyx_csr], ids=["scipy", "cupyx"])
def test_sampler_accepts_sparse(make_csr, sparse_test_data):
    n_errors, n_det, rows, cols, probs = sparse_test_data
    sp = make_csr(n_errors, n_det, rows, cols)

    sampler = BitMatrixSparseSampler(sp, probs, 32, seed=42)
    sampler.sample(32, seed=0)
    outcomes = sampler.get_outcomes()
    assert outcomes.shape[0] == 32


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_runtime_error_before_sample():
    n_errors = 5
    n_results = 3
    check = np.eye(n_errors, n_results, dtype=np.uint8)
    probs = np.full(n_errors, 0.1, dtype=np.float64)

    sampler = BitMatrixSparseSampler(check, probs, 32)
    with pytest.raises(RuntimeError, match="sample\\(\\) has not been called"):
        sampler.get_outcomes()
    with pytest.raises(RuntimeError, match="sample\\(\\) has not been called"):
        sampler.get_errors()


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_seed_reproducibility():
    n_errors = 20
    check = np.eye(n_errors, dtype=np.uint8)
    probs = np.full(n_errors, 0.5, dtype=np.float64)
    shots = 128

    s1 = BitMatrixSparseSampler(check, probs, shots, seed=42)
    s1.sample(shots)
    d1 = s1.get_outcomes(bit_packed=False)

    s2 = BitMatrixSparseSampler(check, probs, shots, seed=42)
    s2.sample(shots)
    d2 = s2.get_outcomes(bit_packed=False)

    np.testing.assert_array_equal(d1, d2)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_consecutive_calls_differ():
    n_errors = 20
    check = np.eye(n_errors, dtype=np.uint8)
    probs = np.full(n_errors, 0.5, dtype=np.float64)
    shots = 128

    sampler = BitMatrixSparseSampler(check, probs, shots, seed=42)

    sampler.sample(shots)
    d1 = sampler.get_outcomes(bit_packed=False).copy()

    sampler.sample(shots)
    d2 = sampler.get_outcomes(bit_packed=False)

    assert not np.array_equal(d1, d2)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_max_shots_exceeded():
    check = np.eye(5, dtype=np.uint8)
    probs = np.full(5, 0.5, dtype=np.float64)
    sampler = BitMatrixSparseSampler(check, probs, 32)
    with pytest.raises(ValueError, match="exceeds max_shots"):
        sampler.sample(64)


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_bitmatrix_sampler_alias():
    assert BitMatrixSampler is BitMatrixSparseSampler


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_parity_invariant():
    """Construct a DEM with known linear dependencies among detectors.

    Errors 0..5 with varying probabilities. Check matrix:
      det0 = err0 ^ err1 ^ err2
      det1 = err2 ^ err3 ^ err4
      det2 = err0 ^ err1 ^ err3 ^ err4       (= det0 ^ det1, err2 cancels)
      det3 = err5                              (independent)
      det4 = err0 ^ err1 ^ err2 ^ err3 ^ err4 ^ err5  (= det0 ^ det1 ^ det3)

    Invariants (exact, every shot):
      det[:,2] == det[:,0] ^ det[:,1]
      det[:,4] == det[:,0] ^ det[:,1] ^ det[:,3]
    """
    n_errors = 6
    n_detectors = 5
    shots = 10000

    check = np.zeros((n_errors, n_detectors), dtype=np.uint8)
    # det0: errors {0, 1, 2}
    check[0, 0] = 1; check[1, 0] = 1; check[2, 0] = 1
    # det1: errors {2, 3, 4}
    check[2, 1] = 1; check[3, 1] = 1; check[4, 1] = 1
    # det2: errors {0, 1, 3, 4} = det0 ^ det1
    check[0, 2] = 1; check[1, 2] = 1; check[3, 2] = 1; check[4, 2] = 1
    # det3: error {5}
    check[5, 3] = 1
    # det4: errors {0, 1, 3, 4, 5} = det0 ^ det1 ^ det3
    check[0, 4] = 1; check[1, 4] = 1
    check[3, 4] = 1; check[4, 4] = 1; check[5, 4] = 1

    probs = np.array([0.1, 0.3, 0.7, 0.9, 0.5, 0.2])

    sampler = BitMatrixSparseSampler(check, probs, shots, options=Options(device_id=0))
    sampler.sample(shots, seed=42)
    det = sampler.get_outcomes(bit_packed=False)

    assert det.shape == (shots, n_detectors)

    parity_01 = det[:, 0] ^ det[:, 1]
    assert np.array_equal(det[:, 2], parity_01), \
        f"det2 != det0^det1: {np.sum(det[:, 2] != parity_01)} violations out of {shots}"

    parity_013 = det[:, 0] ^ det[:, 1] ^ det[:, 3]
    assert np.array_equal(det[:, 4], parity_013), \
        f"det4 != det0^det1^det3: {np.sum(det[:, 4] != parity_013)} violations out of {shots}"

    # Sanity: detectors are not all-zero (probs are non-trivial)
    assert np.any(det[:, 0] != 0), "det0 all zero — sampling may be broken"
    assert np.any(det[:, 1] != 0), "det1 all zero — sampling may be broken"
    assert np.any(det[:, 3] != 0), "det3 all zero — sampling may be broken"


@pytest.mark.skipif(not HAS_CUPY, reason="cupy not available")
def test_deterministic_all_ones():
    """All errors have p=1.0. Detectors with even error count must be 0,
    detectors with odd count must be 1, for every shot."""
    n_errors = 4
    n_detectors = 4
    shots = 100

    check = np.zeros((n_errors, n_detectors), dtype=np.uint8)
    # det0: 2 errors (even) -> always 0
    check[0, 0] = 1; check[1, 0] = 1
    # det1: 3 errors (odd) -> always 1
    check[0, 1] = 1; check[1, 1] = 1; check[2, 1] = 1
    # det2: 1 error (odd) -> always 1
    check[3, 2] = 1
    # det3: 4 errors (even) -> always 0
    check[0, 3] = 1; check[1, 3] = 1; check[2, 3] = 1; check[3, 3] = 1

    probs = np.ones(n_errors)

    sampler = BitMatrixSparseSampler(check, probs, shots, options=Options(device_id=0))
    sampler.sample(shots, seed=0)
    det = sampler.get_outcomes(bit_packed=False)

    expected = np.array([0, 1, 1, 0], dtype=np.uint8)
    for s in range(shots):
        assert np.array_equal(det[s], expected), \
            f"shot {s}: got {det[s]}, expected {expected}"
