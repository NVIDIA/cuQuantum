# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using NumPy ndarrays. Return contraction path and optimizer information.

The contraction result is also a NumPy ndarray.
"""
import numpy as np

from cuquantum import contract


a = np.ones((3,2))
b = np.ones((2,8))
c = np.ones((8,3))

r, (p, i) = contract("ij,jk,kl->il", a, b, c, return_info=True)
print(f"path = {p}")
print(f"optimizer information = {i}")
print(r)

