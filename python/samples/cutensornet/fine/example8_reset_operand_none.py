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
only one set of operands and one contraction at a time.
"""
import logging

import cupy as cp
import numpy as np

import cuquantum

N = 1024

# Turn on logging and set the level to DEBUG to print memory management messages.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Create, prepare, and execute the first iteration on network n1.
a = cp.random.rand(N, N)
b = cp.random.rand(N, N)
n1 = cuquantum.Network("ij,jk", a, b)
n1.contract_path()
r = n1.contract(release_workspace=True)

# Reset network n1 operands to None, and set a and b to None to make memory available for the network n2.
n1.reset_operands(None)
a = b = None

# Create, prepare, and execute the first iteration on network n2.
c = cp.random.rand(N, N)
d = cp.random.rand(N, N)
n2 = cuquantum.Network("ij,jk", c, d)
n2.contract_path()
r = n2.contract(release_workspace=True)

# Reset network n2 operands to None, and set c and d to None to make memory available for next contraction of network n1.
n2.reset_operands(None)
c = d = None

num_iter = 3
# Use the networks as context managers so that internal library resources are properly cleaned up.
with n1, n2:

    for i in range(num_iter):
        print(f"Iteration {i}")
        # Create and set new operands for n1.
        a = cp.random.rand(N, N)
        b = cp.random.rand(N, N)
        n1.reset_operands(a, b)

        # Perform the first contraction, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the second one.
        r = n1.contract(release_workspace=True)

        # Reset network n1 operands to None, and set a and b to None to make memory available for the operands for and contracting network n2.
        n1.reset_operands(None)
        a = b = None

        # Create and set new operands for n2.
        c = cp.random.rand(N, N)
        d = cp.random.rand(N, N)
        n2.reset_operands(c, d)

        # Perform the second contraction, and request that the workspace be released at the end of the operation so that there is enough
        #   memory for the next iteration of the first contraction.
        r = n2.contract(release_workspace=True)

        # Reset network n2 operands to None, and set c and d to None to make memory available for next contraction and operands of network n1.
        n2.reset_operands(None)
        c = d = None
