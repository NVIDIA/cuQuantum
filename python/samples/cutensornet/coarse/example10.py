# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example using PyTorch tensors.

The contraction result is also a PyTorch tensor on the same device.
"""
import torch

from cuquantum import contract, OptimizerOptions


# dev can be any valid device ID on your system, here let's
# pick the first device
dev = 0
a = torch.ones((3,2), device=f'cuda:{dev}')
b = torch.ones((2,3), device=f'cuda:{dev}')

r = contract("ij,jk", a, b)
print(f"result type = {type(r)}")
print(f"result device = {r.device}")
print(r)
