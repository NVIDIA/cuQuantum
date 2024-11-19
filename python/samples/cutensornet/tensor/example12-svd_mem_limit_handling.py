# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SVD Example using CuPy ndarray. Show memory limit handling.

The decomposition results are also CuPy ndarrays.
"""
import cupy as cp

from cuquantum import tensor, MemoryLimitExceeded

# create a random rank-4 tensor
cp.random.seed(2024)
a = cp.random.random((3,2,4,5))

try:
    # use a minimal memory limit to demonstrate the handling of exceeding memory limit
    u, s, v = tensor.decompose('ijab->ijx,xab', a, method=tensor.SVDMethod(), options={'memory_limit': 1})
except MemoryLimitExceeded as e:
    print("handling memory limit...")
    free_memory = cp.cuda.runtime.memGetInfo()[0]
    # setting device memory usage cap to 80% of free memory
    memory_cap = int(free_memory * 0.8)
    print(f"memory cap set to {e.limit} bytes while the required memory is {e.requirement} bytes on device {e.device_id}. (available memory: {memory_cap} bytes)")

    if e.requirement <= memory_cap:
        print(f"memory limit is set to required memory...")
        u, s, v = tensor.decompose('ijab->ijx,xab', a, method=tensor.SVDMethod(), options={'memory_limit': e.requirement})
        print("SVD completed")
    else:
        print("exceeded maximal memory..., skipping SVD")


