# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating a generalized Einstein summation expression.
"""
import numpy as np

from cuquantum import contract


a = np.random.rand(3,2)
b = np.random.rand(3,3)
c = np.random.rand(3,2)
d = np.random.rand(3,4)

# A hyperedge example.
expr = "ij,ik,ij,kl->l"

r = contract(expr, a, b, c, d)
s = np.einsum(expr, a, b, c, d)
assert np.allclose(r, s), "Incorrect results."
