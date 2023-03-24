# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using CuPy ndarray with ellipsis notation.

The decomposition results are also CuPy ndarrays.
"""
import cupy as cp

from cuquantum import tensor


a = cp.ones((3,2,4,5))

q, r = tensor.decompose("ij...->ix,xj...", a)
q1, r1 = tensor.decompose("ijab->ix,xjab", a)

assert q.shape == q1.shape
assert r.shape == r1.shape

