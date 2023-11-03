# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Computing the gradients of a tensor network with CuPy ndarrays.

The gradients are returned as CuPy ndarrays.

This example is also straightforwardly applicable to NumPy ndarrays and PyTorch tensors.
"""

import cupy as cp
import numpy as np

import cuquantum
from cuquantum import cutensornet as cutn


# random-initialize input tensors
a = cp.random.random((3, 4, 5))
b = cp.random.random((4, 5, 6))
c = cp.random.random((6, 5, 2))

# create tensor qualifiers for all input tenors
qualifiers = np.zeros(3, dtype=cutn.tensor_qualifiers_dtype)

# require gradient computation of all inputs
qualifiers[:]['requires_gradient'] = 1

# create a network
tn = cuquantum.Network("abc,bcd,dce->ae", a, b, c, qualifiers=qualifiers)

# perform contraction as usual
# this would prepare the internal cache for gradient computation
tn.contract_path()
out = tn.contract()

# prepare the seed gradient (w.r.t. the output tensor itself, so it's 1)
output_grad = cp.ones_like(out)

# compute the gradients
input_grads = tn.gradients(output_grad)

# the gradient tensors have the same type as the input tensors
assert all(isinstance(arr, cp.ndarray) for arr in input_grads)

# free the network
tn.free()

# check results against PyTorch (if installed)
try:
    import torch
except ImportError:
    torch = None
if torch is not None and torch.cuda.is_available():
    # create torch tenros via zero-copy
    a_t = torch.as_tensor(a, device='cuda')
    b_t = torch.as_tensor(b, device='cuda')
    c_t = torch.as_tensor(c, device='cuda')

    # require gradient computation of all inputs
    a_t.requires_grad_(True)
    b_t.requires_grad_(True)
    c_t.requires_grad_(True)

    # compute the contraction
    out_t = torch.einsum("abc,bcd,dce->ae", a_t, b_t, c_t)

    # backprop to fill the gradients
    output_grad_t = torch.ones_like(out_t)
    out_t.backward(output_grad_t)

    # check results (zero-copy torch tensors as cupy arrays)
    assert cp.allclose(out_t.detach(), out)  # non-leaf nodes need to be detached first
    assert cp.allclose(a_t.grad, input_grads[0])
    assert cp.allclose(b_t.grad, input_grads[1])
    assert cp.allclose(c_t.grad, input_grads[2])
    print("all checked!")
