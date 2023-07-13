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

# bit ordering should only live on host.
bitOrdering  = np.asarray([2, 1, 0], dtype=np.int32)
bitStringLen = bitOrdering.size

# 2 bitStrings are allocated contiguously in single memory chunk.
# Note: bitStrings can also live on the host.
bitStrings   = cp.empty(2, dtype=cp.int64)
bitStrings_res = cp.asarray([0b100, 0b011], dtype=cp.int64)

# In real appliction, random number in range [0, 1) will be used.
# Note: norms can also live on the host.
randnums     = cp.asarray([0.009, 0.5], dtype=cp.float64)

# 2 state vectors are allocated contiguously in single memory chunk.
d_svs        = cp.asarray([[0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j],
                           [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j]], dtype=cp.complex64)

d_svs_res    = cp.asarray([[0.0+0.0j, 0.0+1.0j, 0.0+0.0j, 0.0+0.0j, 
                            0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                           [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
                            0.0+0.0j, 0.0+0.0j, 0.6+0.8j, 0.0+0.0j]], dtype=cp.complex64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# batched measurement
cusv.measure_batched(
    handle, d_svs.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    bitStrings.data.ptr, bitOrdering.ctypes.data, bitStringLen,
    randnums.data.ptr, cusv.Collapse.NORMALIZE_AND_ZERO)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(d_svs_res, d_svs):
    raise ValueError("results mismatch")
if not cp.allclose(bitStrings_res, bitStrings):
    raise ValueError("results mismatch")
print("test passed")
