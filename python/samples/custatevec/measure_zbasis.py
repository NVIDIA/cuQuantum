# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits   = 3
nSvSize      = (1 << nIndexBits)
nBasisBits   = 3
basisBits    = np.asarray([0, 1, 2], dtype=np.int32)

# In real appliction, random number in range [0, 1) will be used.
randnum      = 0.2

h_sv         = np.asarray([0.0+0.0j, 0.0+0.1j, 0.3+0.4j, 0.1+0.2j, 
                           0.2+0.2j, 0.3+0.3j, 0.1+0.1j, 0.4+0.5j], dtype=np.complex64)
d_sv         = cp.asarray(h_sv)

expected_sv  = np.asarray([0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.2+0.4j, 
                           0.0+0.0j, 0.6+0.6j, 0.2+0.2j, 0.0+0.0j], dtype=np.complex64)
expected_parity = 0

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# measurement on z basis
parity = cusv.measure_on_z_basis(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    basisBits.ctypes.data, nBasisBits, randnum, cusv.Collapse.NORMALIZE_AND_ZERO)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(expected_sv, d_sv):
    raise ValueError("results mismatch")
if expected_parity != parity:
    raise ValueError("results mismatch")
print("test passed")
