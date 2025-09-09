# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using NumPy ndarray. Specify the CUDA stream for the computation.

When NumPy operands are used, the stream must be a pointer to a CUDA stream or a cuda.core.Stream object.

The decomposition results are also NumPy ndarrays.
"""
import numpy as np
import cuda.core.experimental as ccx

from cuquantum.tensornet import tensor

d0 = ccx.Device(0)
d0.set_current()
s = d0.create_stream()

a = np.ones((3,2,4,5))

q, r = tensor.decompose("ijab->ixa,xbj", a, stream=s)
print(q)
print(r)

