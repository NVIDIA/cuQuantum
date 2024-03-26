# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This example shows how to manage memory resources used by stateful objects. This is useful when the tensor network
needs a lot of memory and calls to execution methods such as autotuning, contract, and gradients on a
stateful object are interleaved with calls to other operations (including on other tensor networks) also requiring
a lot of memory.

In this example, we use two tensor networks representing two large matrix multiplications and perform the two
contractions in a loop in an interleaved manner. We assume that the available device memory is large enough for
only one contraction at a time.
"""
import logging

import cupy as cp
import numpy as np

import cuquantum

# Create operands for an MM.
N = 1024
a = cp.random.rand(N, N)
b = cp.random.rand(N, N)
c = cp.random.rand(N, N)
d = cp.random.rand(N, N)

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Create and prepare network n1.
n1 = cuquantum.Network("ij,jk", a, b)
n1.contract_path()

# Create and prepare network n2.
n2 = cuquantum.Network("ij,jk", c, d)
n2.contract_path()

num_iter = 3
# Use the networks as context managers so that internal library resources are properly cleaned up.
with n1, n2:

    for i in range(num_iter):
        print(f"Iteration {i}")
        # Perform the first contraction, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the second one.
        r = n1.contract(release_workspace=True)

        # Update n1's operands for the next iteration.
        if i < num_iter-1:
            a[:] = cp.random.rand(N, N)
            b[:] = cp.random.rand(N, N)

        # Perform the second contraction, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the next iteration of the first contraction.
        r = n2.contract(release_workspace=True)

        # Update n2's operands for the next iteration.
        if i < num_iter-1:
            c[:] = cp.random.rand(N, N)
            d[:] = cp.random.rand(N, N)
