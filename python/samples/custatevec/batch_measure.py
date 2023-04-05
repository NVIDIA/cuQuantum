# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits   = 3
nSvSize      = (1 << nIndexBits)
bitStringLen = 3
bitOrdering  = np.asarray([2, 1, 0], dtype=np.int32)

# In real appliction, random number in range [0, 1) will be used.
randnum      = 0.5

h_sv         = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                           0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
d_sv         = cp.asarray(h_sv)

expected_sv  = np.asarray([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 
                           0.0+0.0j, 0.0+0.0j, 0.6+0.8j, 0.0+0.0j], dtype=np.complex64)
expected_bitString = np.asarray([1, 1, 0], dtype=np.int32)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# allocate host memory to hold the result
bitString = np.empty((bitStringLen,), dtype=np.int32)

# batch measurement
cusv.batch_measure(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, bitString.ctypes.data,
    bitOrdering.ctypes.data, bitStringLen, randnum, cusv.Collapse.NORMALIZE_AND_ZERO)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(expected_sv, d_sv):
    raise ValueError("results mismatch")
if not np.allclose(expected_bitString, bitString):
    raise ValueError("results mismatch")
print("test passed")
