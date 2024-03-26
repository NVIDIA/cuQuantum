# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This example shows how to manage memory resources used by stateful objects. This is useful when the tensor network
needs a lot of memory and calls to execution methods such as autotuning, contract, and gradients on a
stateful object are interleaved with calls to other operations (including on other tensor networks) also requiring
a lot of memory.

We contract and find the gradients of a tensor network representing a matrix multiplication, then update the operands
based on the gradients (simulating updating weights), before moving on to subsequent iterations of contraction and
gradient calculation. For the purposes of this example, we assume that the available device memory is not large enough
for updating the operands if the network doesn't release the workspace memory used for the gradient calculation.
"""
import logging

import cupy as cp
import numpy as np

import cuquantum

# Create operands for an MM.
N = 1024
a = cp.random.rand(N, N)
b = cp.random.rand(N, N)

# Create qualifiers and indicate that gradients are required.
q = np.zeros((2,), dtype=cuquantum.cutensornet.tensor_qualifiers_dtype)
q['requires_gradient'] = True

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Create the network.
with cuquantum.Network("ij,jk", a, b, qualifiers=q) as n:

    # Compute the path.
    n.contract_path()

    for i in range(2):
        print(f"Iteration {i}")
        # Perform the contraction.
        r = n.contract()

        # Find the gradients, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the following calculation updating the operands.
        c = cp.random.rand(N, N)
        g = n.gradients(c, release_workspace=True)

        # Update the operands based on some function of the gradients. Here we update in-place so we don't need to call reset_operands().
        a[:] += 0.1 * g[0]
        b[:] += 0.1 * g[1]
