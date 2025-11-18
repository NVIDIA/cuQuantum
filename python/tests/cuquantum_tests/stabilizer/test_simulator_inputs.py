# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for package semantics and edge cases in custabilizer pythonic API."""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from cuquantum.stabilizer import Circuit, FrameSimulator, Options

packages = ["numpy", "cupy"]


@pytest.fixture(params=packages)
def package(request):
    if not HAS_CUPY and request.param == "cupy":
        pytest.skip("CuPy not available")
    return request.param


# -- < Odd number of shots > --


@pytest.mark.parametrize("num_shots", [1, 15, 31, 33, 129, 1023, 2049])
def test_odd_shots_packed(num_shots):
    """Test simulator with odd numbers of shots (not multiple of 32)."""
    sim = FrameSimulator(2, num_shots, num_measurements=1)
    assert sim.num_paulis == num_shots

    circ = Circuit("H 0\nCNOT 0 1\nM 0")
    sim.apply(circ)

    # Test that we can retrieve results
    x, z = sim.get_pauli_xz_bits(bit_packed=False)
    assert x.shape == (2, num_shots)
    assert z.shape == (2, num_shots)

    m = sim.get_measurement_bits(bit_packed=False)
    assert m.shape == (1, num_shots)

    # Test bit-packed format
    x_packed, z_packed = sim.get_pauli_xz_bits(bit_packed=True)
    expected_stride = ((num_shots + 31) // 32) * 4
    assert x_packed.shape == (2, expected_stride)
    assert z_packed.shape == (2, expected_stride)


@pytest.mark.parametrize("num_shots", [1, 3, 137, 1024 + 137])
def test_odd_shots_unpacked(num_shots, package):
    """Test simulator with odd numbers of shots using cupy."""
    sim = FrameSimulator(2, num_shots, num_measurements=1, package=package)
    assert sim.num_paulis == num_shots

    circ = Circuit("H 0\nCNOT 0 1\nM 0")
    sim.apply(circ)

    x, z = sim.get_pauli_xz_bits(bit_packed=False)
    m = sim.get_measurement_bits(bit_packed=False)
    cls_ = cp.ndarray if package == "cupy" else np.ndarray
    assert isinstance(x, cls_)
    assert isinstance(z, cls_)
    assert isinstance(m, cls_)
    assert x.shape == (2, num_shots)
    assert z.shape == (2, num_shots)
    assert m.shape == (1, num_shots)


# -- </ Odd number of shots > --


def test_package_semantics_no_inputs_kwarg(package):
    """
    No inputs provided, package kwarg provided -> package output.
    """
    sim = FrameSimulator(2, 64, num_measurements=5, package=package)
    circ = Circuit("X_ERROR(1) 0\nDEPOLARIZE2(1) 1 0\nM 0 1")
    sim.apply(circ)
    x, z = sim.get_pauli_xz_bits(bit_packed=False)
    m = sim.get_measurement_bits(bit_packed=True)
    cls_ = cp.ndarray if package == "cupy" else np.ndarray
    assert isinstance(x, cls_)
    assert isinstance(z, cls_)
    assert isinstance(m, cls_)


# -- < Input tables > --


def test_simulator_input_tables_packed(package):
    """Test package semantics: numpy input -> numpy output."""
    xp_ = cp if package == "cupy" else np
    num_qubits = 2
    num_shots = 64
    stride = ((num_shots + 31) // 32) * 4
    x_table = xp_.zeros((num_qubits, stride), dtype=xp_.uint8)
    z_table = xp_.zeros((num_qubits, stride), dtype=xp_.uint8)

    sim = FrameSimulator(
        num_qubits,
        num_shots,
        num_measurements=3,
        x_table=x_table,
        z_table=z_table,
        bit_packed=True,
    )

    # circ = Circuit("X_ERROR(1) 0\nZ_ERROR(1) 1\nH 0 1\nCNOT 1 2\nM 0 1 2")
    circ = Circuit("X_ERROR(1) 0\nZ_ERROR(1) 1\nM 0 1")
    sim.apply(circ)
    x, z = sim.get_pauli_xz_bits(bit_packed=True)
    m = sim.get_measurement_bits(bit_packed=False)

    assert isinstance(x, xp_.ndarray)
    assert isinstance(z, xp_.ndarray)
    assert isinstance(m, xp_.ndarray)
    assert not xp_.all(x == 0)
    assert not xp_.all(z == 0)
    if package == "cupy":
        assert z.data.ptr == z_table.data.ptr, "Z table should be attached, not copied"
        assert x.data.ptr == x_table.data.ptr, "X table should be attached, not copied"


def test_simulator_input_tables_unpacked(package):
    """Test creating simulator with cupy input tables (unpacked)."""
    num_qubits = 3
    num_shots = 1025
    num_measurements = 1

    xp_ = cp if package == "cupy" else np
    x_table = xp_.zeros((num_qubits, num_shots), dtype=xp_.uint8)
    x_table[1, :] = 1
    z_table = xp_.zeros((num_qubits, num_shots), dtype=xp_.uint8)

    sim = FrameSimulator(
        num_qubits,
        num_shots,
        num_measurements=num_measurements,
        x_table=x_table,
        z_table=z_table,
        bit_packed=False,
        package="numpy", # data package has higher priority than package kwarg
    )

    assert sim.num_qubits == num_qubits
    assert sim.num_paulis == num_shots

    # Apply circuit and verify output is cupy
    circ = Circuit("H 1\nZ_ERROR(1) 0\nM 2")
    sim.apply(circ)

    x, z = sim.get_pauli_xz_bits(bit_packed=False)
    assert isinstance(x, xp_.ndarray)
    assert isinstance(z, xp_.ndarray)
    assert x.shape == (num_qubits, num_shots)
    assert z.shape == (num_qubits, num_shots)
    assert np.all(x[1, :] == 0)
    assert np.all(z[1, :] == 1)
    assert np.all(z[0, :] == 1)

    m = sim.get_measurement_bits(bit_packed=False)
    assert isinstance(m, xp_.ndarray)
    assert m.shape == (num_measurements, num_shots)


def test_simulator_set_input_tables_packed(package):
    """Test set_input_tables with numpy arrays."""
    num_qubits = 2
    num_shots = 64
    num_measurements = 2

    sim = FrameSimulator(num_qubits, num_shots, num_measurements=num_measurements)

    xp_ = cp if package == "cupy" else np
    stride = ((num_shots + 31) // 32) * 4
    x_table = xp_.zeros((num_qubits, stride), dtype=xp_.uint8)
    z_table = xp_.zeros((num_qubits, stride), dtype=xp_.uint8)
    m_table = xp_.zeros((num_measurements, stride), dtype=xp_.uint8)

    sim.set_input_tables(x=x_table, z=z_table, m=m_table, bit_packed=True)

    # Apply circuit
    circ = Circuit("X_ERROR(1) 0\nZ_ERROR(1) 1\nM 0 1")
    sim.apply(circ)

    # Verify output type matches input
    x, z = sim.get_pauli_xz_bits(bit_packed=True)
    assert isinstance(x, xp_.ndarray)
    assert isinstance(z, xp_.ndarray)
    assert x.shape == (num_qubits, stride)
    assert z.shape == (num_qubits, stride)

    m = sim.get_measurement_bits(bit_packed=True)
    assert isinstance(m, xp_.ndarray)
    assert m.shape == (num_measurements, stride)
    if package == "cupy":
        assert z.data.ptr == z_table.data.ptr, "Z table should be attached to simulator"
        assert x.data.ptr == x_table.data.ptr, "X table should be attached to simulator"
        assert m.data.ptr == m_table.data.ptr, "M table should be attached to simulator"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
