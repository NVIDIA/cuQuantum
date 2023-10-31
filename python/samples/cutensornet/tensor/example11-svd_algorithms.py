# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
truncated SVD Example using CuPy ndarray with various SVD algorithms.

The decomposition results are also CuPy ndarrays.
"""
import cupy as cp

from cuquantum import tensor

a = cp.ones((3,2,4,5))

base_options = {'max_extent': 4,
                'abs_cutoff': 0.1,
                'rel_cutoff': 0.1}


for algorithm in ('gesvd', 'gesvdj', 'gesvdr', 'gesvdp'):
    method = tensor.SVDMethod(algorithm=algorithm, **base_options)
    u, s, v, info = tensor.decompose("ijab->ixa,xbj", a, method=method, return_info=True)
    print(s)
    print(info)

