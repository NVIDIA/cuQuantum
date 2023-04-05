# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example illustrating ellipsis broadcasting.
"""
import numpy as np

from cuquantum import contract


a = np.arange(3.).reshape(3,1)
b = np.arange(9.).reshape(3,3)

# Double inner product (Frobenuis inner product) of two matrices. 
expr = "...,...->"

r = contract(expr, a, b)
print(r)
assert np.allclose(r, 54.), "Incorrect results."
