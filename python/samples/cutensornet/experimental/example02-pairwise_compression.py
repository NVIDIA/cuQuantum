# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example of pairwise tensor compression with contract_decompose

NumPy ndarrays are used as inputs.
"""
import numpy as np

from cuquantum import contract
from cuquantum.cutensornet.experimental import contract_decompose


a = np.ones((2,2,2))
b = np.ones((2,2,2))

# use SVD to compress two tensors:
#   i     k     m                        i     k     m
# =====A=====B=====      ===>          =====A-----B=====       
#      |j   l|                              |j   l|

abs_cutoff = 1e-12
compress_algorithm = {
    'qr_method' : False,
    'svd_method': {'abs_cutoff':abs_cutoff, 'partition': 'UV'} # singular values are partitioned onto A/B equally
}

# compare the difference after compression
a_svd, _, b_svd = contract_decompose('ijk,klm->ijk,klm', a, b, algorithm=compress_algorithm)
diff = contract('ijk,klm', a, b) - contract('ijk,klm', a_svd, b_svd)

print(f"After compression with cutoff {abs_cutoff}")
print(f"    Shape of A, B: {a_svd.shape} {b_svd.shape}")
print(f"    Maxdiff error: {abs(diff).max()}")