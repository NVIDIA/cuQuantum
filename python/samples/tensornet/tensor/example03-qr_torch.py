# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
QR Example using PyTorch Tensor.

The decomposition results are also PyTorch Tensors.
"""
import torch

from cuquantum.tensornet import tensor


a = torch.ones((3,2,4,5))

q, r = tensor.decompose("ijab->ixa,xbj", a)
print(q)
print(r)

