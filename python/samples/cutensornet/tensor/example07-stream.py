# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using NumPy ndarray. Specify the CUDA stream for the computation

The decomposition results are also NumPy ndarrays.
"""
import numpy as np
import cupy as cp

from cuquantum import tensor


a = np.ones((3,2,4,5))

stream = cp.cuda.Stream()
q, r = tensor.decompose("ijab->ixa,xbj", a, stream=stream)
print(q)
print(r)

