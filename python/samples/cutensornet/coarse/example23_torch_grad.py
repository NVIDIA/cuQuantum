# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example: Demo using cuquantum.contract() in a PyTorch compute graph out of box.

This sample requires PyTorch.
"""

import torch

import cuquantum
from cuquantum import cutensornet as cutn


# random-initialize input tensors on GPU, and require gradient computation of all inputs
kwargs = {'device': 'cuda',
          'requires_grad': True,
          'dtype': torch.complex128}
a = torch.rand((3, 4), **kwargs)
b = torch.rand((4, 5, 6, 3), **kwargs)
c = torch.rand((3, 3), **kwargs)

# create a hypothetical workload using PyTorch operators
def compute(func, expr, a, b, c):
    # note: cannot perform in-place ops on leaf nodes
    a = a * a
    b = -10 + b
    c = torch.cos(c)
    d = func(expr, a, b, c)
    return torch.sum(d, dim=0, keepdim=True)

# use cuquantum.contract() in the workload to compute gradients
out_cuqnt = compute(cuquantum.contract, "ab,bcde,ef->acdf", a, b, c)

# backprop to fill the gradients
output_grad = torch.ones_like(out_cuqnt)
out_cuqnt.backward(output_grad)

# store the computed gradients for later verification
input_grads_cuqnt = (a.grad, b.grad, c.grad)

# now let's reset the gradients and redo the computation using
# torch.einsum() for comparison
a.grad, b.grad, c.grad = None, None, None
out_torch = compute(torch.einsum, "ab,bcde,ef->acdf", a, b, c)
out_torch.backward(output_grad)
input_grads_torch = (a.grad, b.grad, c.grad)

# check results
assert all(
    torch.allclose(grad_cuqnt, grad_torch)
    for grad_cuqnt, grad_torch in zip(input_grads_cuqnt, input_grads_torch)
)
print("all checked!")
