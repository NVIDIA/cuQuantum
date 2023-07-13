# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Verify FLOPS and largest intermediate size against NumPy for a given path.
"""
import re

from cuquantum import contract_path, OptimizerOptions
import numpy as np

expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]

operands = [np.random.rand(*shape) for shape in shapes]

# NumPy path and metrics.
path_np, i = np.einsum_path(expr, *operands)

flops_np = float(re.search("Optimized FLOP count:(.*)\n", i).group(1))
largest_np = float(re.search("Largest intermediate:(.*) elements\n", i).group(1))
flops_np -= 1    # NumPy adds 1 to the FLOP count.

# Set path and obtain metrics.
o = OptimizerOptions(path=path_np[1:])
path, i = contract_path(expr, *operands, optimize=o)
assert list(path) == path_np[1:], "Error: path doesn't match what was set."

flops = i.opt_cost
largest = i.largest_intermediate

if flops != flops_np or largest != largest_np:
    message = f""" Results don't match.
path = {path_np}
flops: NumPy = {flops_np}, cuTensorNet = {flops},
largest intermediate: NumPy  = {largest_np}, cuTensorNet = {largest}
"""
    raise ValueError(message)
