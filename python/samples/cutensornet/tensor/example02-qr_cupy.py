# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using CuPy ndarray.

The decomposition results are also CuPy ndarrays.
"""
import cupy as cp

from cuquantum import tensor


a = cp.ones((3,2,4,5))

q, r = tensor.decompose("ijab->ixa,xbj", a)
print(q)
print(r)

