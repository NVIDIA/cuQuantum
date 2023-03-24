# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example of contract and QR decompose an arbitrary network

NumPy ndarrays are used as inputs.
"""
import numpy as np

from cuquantum import contract, tensor
from cuquantum.cutensornet.experimental import contract_decompose


inputs = ('ab', 'bcd', 'cde', 'exg', 'ayg')
outputs = ('xz', 'zy')

# creating random input tensors
np.random.seed(0)
size_dict = {}
operands = []
for modes in inputs:
    shape = []
    for m in modes:
        if m not in size_dict:
            size_dict[m] = np.random.randint(2,6)
        shape.append(size_dict[m])
    operands.append(np.random.random(shape))

subscripts = ",".join(inputs) + "->" + ",".join(outputs)

# contraction followed by QR decomposition
algorithm = {'qr_method':{}, 'svd_method':False}

q, r = contract_decompose(subscripts, *operands, algorithm=algorithm)

# compute the full network contraction using the original input operands
result = contract(",".join(inputs), *operands)
# compute the full network contraction using the decomposed outputs
result_reference = contract(",".join(outputs), q, r)

diff = abs(result - result_reference).max()
print(f"After contract and QR decomposition")
print(f"max diff={diff}")
