# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays. Specify network options.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract, NetworkOptions


a = np.ones((3,2))
b = np.ones((2,3))

o = NetworkOptions(memory_limit="10kb")    # As a value with units.
o = NetworkOptions(memory_limit=12345)     # As a number of bytes (int or float).
o = NetworkOptions(memory_limit="10%")     # As a percentage of device memory.

r = contract("ij,jk", a, b, options=o)
print(r)

