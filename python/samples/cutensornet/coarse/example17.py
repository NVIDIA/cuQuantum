# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating mode broadcasting.
"""
import numpy as np

from cuquantum import contract


a = np.random.rand(3,1)
b = np.random.rand(3,3)

expr = "ij,jk"

r = contract(expr, a, b)
s = np.einsum(expr, a, b)
assert np.allclose(r, s), "Incorrect results."
