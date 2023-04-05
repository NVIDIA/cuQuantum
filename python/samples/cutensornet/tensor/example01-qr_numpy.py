# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using NumPy ndarray.

The decomposition results are also NumPy ndarrays.
"""
import numpy as np

from cuquantum import tensor


a = np.ones((3,2,4,5))

q, r = tensor.decompose("ijab->ixa,xbj", a)
print(q)
print(r)

