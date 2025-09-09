# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays. Specify CUDA stream for the computation.

When NumPy operands are used, the stream must be a pointer to a CUDA stream or a cuda.core.Stream object.

The contraction result is also a NumPy ndarray.
"""
import numpy as np
import cuda.core.experimental as ccx

from cuquantum.tensornet import contract

d0 = ccx.Device(0)
d0.set_current()
s = d0.create_stream()

a = np.ones((3,2))
b = np.ones((2,3))

r = contract("αβ,βγ->αγ", a, b, stream=s)
print(r)

