# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""FrameSimulator functionality test of cuStabilizer python bindings."""

import cupy as cp
import numpy as np
from cuquantum.bindings import custabilizer as cust


def test_frame_simulator():
  handle = cust.create()
  assert handle != 0, "Failed to create handle"

  # Create a simple circuit
  circuit_string = "H 0\nCNOT 0 1\nM 0 1"
  buffer_size = cust.circuit_size_from_string(handle, circuit_string)
  assert buffer_size > 0, "Failed to get circuit size"
  
  device_buffer = cp.cuda.alloc(buffer_size)
  assert device_buffer.ptr != 0, "Failed to create buffer"
  
  circuit = cust.create_circuit_from_string(handle, circuit_string, device_buffer.ptr, buffer_size)
  assert circuit != 0, "Failed to create circuit on device"

  # Set up frame simulator parameters
  num_qubits = 3
  num_shots = 32
  num_measurements = 2
  table_stride_major = ((num_shots + 31) // 32) * 4
  
  # Create frame simulator
  frame_simulator = cust.create_frame_simulator(handle, num_qubits, num_shots, num_measurements, table_stride_major)
  assert frame_simulator != 0, "Failed to create frame simulator"

  # Allocate and initialize bit tables on host
  bit_table_size = num_qubits * table_stride_major
  m_table_size = num_measurements * table_stride_major
  
  x_table_host = np.zeros(bit_table_size // 4, dtype=np.uint32)
  z_table_host = np.zeros(bit_table_size // 4, dtype=np.uint32)
  m_table_host = np.zeros(m_table_size // 4, dtype=np.uint32)
  
  # Initialize with example Pauli frame: XYZ on first 3 qubits
  # Qubit 0: X -> x_table[0] |= 0x1, z_table[0] &= ~0x1
  # Qubit 1: Y -> x_table[1] |= 0x1, z_table[1] |= 0x1
  # Qubit 2: Z -> x_table[2] &= ~0x1, z_table[2] |= 0x1
  x_table_host[0] = 0x00000001
  x_table_host[1] = 0x00000001
  x_table_host[2] = 0x00000000
  
  z_table_host[0] = 0x00000000
  z_table_host[1] = 0x00000001
  z_table_host[2] = 0x00000001
  
  # Copy tables to device
  x_table_d = cp.asarray(x_table_host)
  z_table_d = cp.asarray(z_table_host)
  m_table_d = cp.asarray(m_table_host)
  
  # Apply circuit
  seed = 5
  randomize_frame_after_measurement = 1
  cust.frame_simulator_apply_circuit(handle, frame_simulator, circuit, randomize_frame_after_measurement, seed,
                                     x_table_d.data.ptr, z_table_d.data.ptr, m_table_d.data.ptr, 0)
  
  # Copy results back
  x_table_result = cp.asnumpy(x_table_d)
  z_table_result = cp.asnumpy(z_table_d)
  m_table_result = cp.asnumpy(m_table_d)
  
  assert x_table_result is not None, "Failed to get x_table results"
  assert z_table_result is not None, "Failed to get z_table results"
  assert m_table_result is not None, "Failed to get m_table results"

  # Clean up
  cust.destroy_frame_simulator(frame_simulator)
  cust.destroy_circuit(circuit)
  cust.destroy(handle)


if __name__ == "__main__":
  test_frame_simulator()
