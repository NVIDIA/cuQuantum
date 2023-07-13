# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType


nSVs         = 2
nIndexBits   = 3
svStride     = (1 << nIndexBits)

# square absolute values of state vector elements for 0/2-th bits will be summed up
# bit ordering should only live on host.
bitOrdering  = np.asarray([1], dtype=np.int32)
bitStringLen = bitOrdering.size

# 2 state vectors are allocated contiguously in single memory chunk.
d_svs        = cp.asarray([[0.0  + 0.0j,  0.0  + 0.1j,  0.1  + 0.1j,  0.1  + 0.2j, 
                            0.2  + 0.2j,  0.3  + 0.3j,  0.3  + 0.4j,  0.4  + 0.5j],
                           [0.25 + 0.25j, 0.25 + 0.25j, 0.25 + 0.25j, 0.25 + 0.25j,
                            0.25 + 0.25j, 0.25 + 0.25j, 0.25 + 0.25j, 0.25 + 0.25j]], dtype=cp.complex64)

abs2sumStride = 2
batchedAbs2sumSize = nSVs * abs2sumStride

# abs2sum arrays are allocated contiguously in single memory chunk
# Note: abs2sum can also live on the host.
abs2sum = cp.empty(batchedAbs2sumSize, dtype=cp.float64)
abs2sum_res = cp.asarray([0.27, 0.73, 0.5, 0.5], dtype=cp.float64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# compute abs2sum arrays
cusv.abs2sum_array_batched(
    handle, d_svs.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    abs2sum.data.ptr, abs2sumStride,
    bitOrdering.ctypes.data, bitStringLen, 0, 0, 0)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(abs2sum_res, abs2sum):
    raise ValueError("results mismatch")
print("test passed")
