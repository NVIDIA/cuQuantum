# Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cuda.bindings.runtime as cudart
from cuquantum.bindings import custabilizer as custab


def test_version():
    v = custab.get_version()
    assert v is not None
    assert v > 0
    print(f"cuStabilizer version: {v}")

def test_circuit():
  handle = custab.create();
  assert handle != 0, "Failed to create handle"

  buffer_size = 0;
  circuit_string = "H 0\nCNOT 0 1\nREPEAT 2 {\nZ_ERROR(0.1) 2\nREPEAT 3 {\nCNOT 0 2\nM 0 2\n}\n}\nM 0 1 2"

  buffer_size = custab.circuit_size_from_string(handle, circuit_string);
  assert buffer_size > 0, "Failed to get circuit size"
  #  create gpu buffer to pass to function
  err, device_ptr = cudart.cudaMalloc(buffer_size)
  assert err == 0, "Failed to create buffer"
  circuit = custab.create_circuit_from_string(handle, circuit_string, device_ptr, buffer_size)
  assert circuit != 0, "Failed to create circuit on device"
  custab.destroy_circuit(circuit)
  custab.destroy(handle)

if __name__ == "__main__":
  test_version()
  test_circuit()
  print("All bindings tests passed!")
