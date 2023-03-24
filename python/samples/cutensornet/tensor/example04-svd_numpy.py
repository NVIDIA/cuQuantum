# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SVD Example using NumPy ndarray.

The decomposition results are also NumPy ndarrays.
"""
import numpy as np

from cuquantum import tensor


a = np.ones((3,2,4,5))

u, s, v = tensor.decompose("ijab->ixa,xbj", a, method=tensor.SVDMethod())
print(s)

