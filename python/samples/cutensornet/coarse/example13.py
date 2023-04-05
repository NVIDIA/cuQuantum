# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Set sliced modes.
"""
import re

from cuquantum import contract, OptimizerOptions
import numpy as np

expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

operands = [np.random.rand(*shape) for shape in shapes]

# Set sliced modes.
o = OptimizerOptions(slicing=(('e', 2), ('h',1)))

r = contract(expr, *operands, optimize=o)
s = np.einsum(expr, *operands)
assert np.allclose(r, s), "Incorrect results."
