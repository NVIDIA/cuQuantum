# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using CuPy ndarray. Specify the CUDA stream for the computation.

When CuPy operands are used, the stream must be a pointer to a CUDA stream or a cupy.cuda.Stream object.

The decomposition results are also CuPy ndarrays.
"""
import cupy as cp

from cuquantum.tensornet import tensor

s = cp.cuda.Stream()

a = cp.ones((3,2,4,5))

q, r = tensor.decompose("ijab->ixa,xbj", a, stream=s)
s.synchronize()
print(q)
print(r)

