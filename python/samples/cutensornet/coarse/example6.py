# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays. Specify CUDA stream for the computation.

The contraction result is also a NumPy ndarray.
"""
import cupy as cp
import numpy as np

from cuquantum import contract, NetworkOptions


a = np.ones((3,2))
b = np.ones((2,3))

s = cp.cuda.Stream()
r = contract("αβ,βγ->αγ", a, b, stream=s)
print(r)

