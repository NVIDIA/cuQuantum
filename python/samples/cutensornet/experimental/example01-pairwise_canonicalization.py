# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example of pairwise tensor canonicalization with contract_decompose

CuPy ndarrays are used as inputs.
"""
import cupy as cp

from cuquantum import contract
from cuquantum.cutensornet.experimental import contract_decompose

a = cp.ones((2,2,2))
b = cp.ones((2,2,2))

# use QR to canonicalize two tensors:
#   i     k     m                        i     k     m
# =====A=====B=====      ===>          =====A---->B=====       
#      |j   l|                              |j   l|

canonicalize_algorithm = {
    'qr_method': {}, 
    'svd_method': False
}

a_qr, b_qr = contract_decompose('ijk,klm->ijk,klm', a, b, algorithm=canonicalize_algorithm)
# compare the difference after canonicalization
out1 = contract('ijk,klm', a, b)
out2 = contract('ijk,klm', a_qr, b_qr)

assert cp.allclose(out1, out2)

print("After canonicalization")
print(f"    Shape of A, B: {a_qr.shape} {b_qr.shape}")
