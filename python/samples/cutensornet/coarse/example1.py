# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays with explicit Einstein summation.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract


a = np.ones((3,2))
b = np.ones((2,3))

r = contract("ij,jk->ik", a, b)
print(r)

