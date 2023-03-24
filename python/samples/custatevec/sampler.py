# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)
nMaxShots  = 5
nShots     = 5

bitStringLen  = 2;
bitOrdering   = np.asarray([0, 1], dtype=np.int32)

bitStrings = np.empty((nShots,), dtype=np.int64)
bitStrings_expected = np.asarray([0b00, 0b01, 0b10, 0b11, 0b11], dtype=np.int64)

h_sv          = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)

d_sv = cp.asarray(h_sv)

# In real appliction, random numbers in range [0, 1) will be used.
randnums      = np.asarray([0.1, 0.8, 0.4, 0.6, 0.2], dtype=np.float64)

########################################################################

# cuStateVec handle initialization
handle = cusv.create()

# create sampler and check the size of external workspace
sampler, extraWorkspaceSizeInBytes = cusv.sampler_create(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, nMaxShots)

# allocate external workspace
extraWorkspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)

# sample preprocess
cusv.sampler_preprocess(
    handle, sampler, extraWorkspace.ptr, extraWorkspaceSizeInBytes)

# sample bit strings
cusv.sampler_sample(
    handle, sampler, bitStrings.ctypes.data, bitOrdering.ctypes.data, bitStringLen,
    randnums.ctypes.data, nShots, cusv.SamplerOutput.ASCENDING_ORDER)

# destroy sampler
cusv.sampler_destroy(sampler)

# destroy handle
cusv.destroy(handle)

if not np.allclose(bitStrings, bitStrings_expected):
    raise ValueError("results mismatch")
print("test passed")
