# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example of applying gate operand to two connecting tensors with contract_decompose

NumPy ndarrays are used as inputs.
"""
import numpy as np

from cuquantum import contract
from cuquantum.cutensornet.experimental import contract_decompose


a = np.ones((2,2,2))
b = np.ones((2,2,2))
gate = np.ones((2,2,2,2))

# absorb the gate tensor onto two connecting tensors

#   i  |  k  |   m                        
# =====A=====B=====                  i     k     m
#      |j   l|           ===>      =====A-----B=====               
#      GGGGGGG                         p|    q|
#      |p   q|

abs_cutoff = 1e-12

# use QR to assist in contraction decomposition
# note this is currently only supported for fully connected network with three tensors
gate_algorithm = {
    'qr_method' : {},
    'svd_method': {'abs_cutoff':abs_cutoff, 'partition': 'UV'} # singular values are partitioned onto A/B equally
}

# compare the difference after compression
a_svd, _, b_svd = contract_decompose('ijk,klm,jlpq->ipk,kqm', a, b, gate, algorithm=gate_algorithm)
diff = contract('ijk,klm,ijpq', a, b, gate) - contract('ipk,kqm', a_svd, b_svd)

print(f"After compression with cutoff {abs_cutoff}")
print(f"    Shape of A, B: {a_svd.shape} {b_svd.shape}")
print(f"    Maxdiff error: {abs(diff).max()}")