# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SVD Example using PyTorch Tensor.

The decomposition results are also PyTorch Tensors.
"""
import torch

from cuquantum import tensor


a = torch.ones((3,2,4,5))

u, s, v = tensor.decompose("ijab->ixa,xbj", a, method=tensor.SVDMethod())
print(s)

