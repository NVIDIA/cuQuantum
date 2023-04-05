# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating a generalized Einstein summation expression.
"""
import numpy as np

from cuquantum import contract


a = np.arange(16.).reshape(4,4)
b = np.arange(64.).reshape(4,4,4)

# Elementwise multiplication of tensor diagonals.
expr = "ii,iii->i"

r = contract(expr, a, b)
s = np.einsum(expr, a, b)
assert np.allclose(r, s), "Incorrect results."
