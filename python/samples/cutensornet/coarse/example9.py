# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using CuPy ndarrays. 

The contraction result is also a CuPy ndarray on the same device.
"""
import cupy as cp

from cuquantum import contract, OptimizerOptions


# dev can be any valid device ID on your system, here let's
# pick the first device
dev = 0
with cp.cuda.Device(dev):
    a = cp.ones((3,2))
    b = cp.ones((2,3))

r = contract("ij,jk", a, b)
print(f"result type = {type(r)}")
print(f"result device = {r.device}")
print(r)
