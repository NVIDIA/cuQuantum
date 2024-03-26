# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


nSVs          = 2
nIndexBits    = 3
svSize        = (1 << nIndexBits)
svStride      = svSize

basisBits     = [2]
nBasisBits    = len(basisBits)

nMatrices     = 2
exp_values = np.empty(nMatrices * nSVs, dtype=np.complex128)
expected   = np.asarray([0.48, -0.10, 0.46, -0.16], dtype=np.complex128)

# 2 state vectors are allocated contiguously in single memory chunk.
d_svs         = cp.asarray([[0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                             0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j],
                            [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                             0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.5+0.4j]], dtype=cp.complex64)

# 2 gate matrices are allocated contiguously in single memory chunk.
# Note: gate matrices can also live on the host.
d_matrices    = cp.asarray([[0.0+0.0j, 1.0+0.0j, 
                             1.0+0.0j, 0.0+0.0j],
                            [0.0+0.0j, 0.0-1.0j, 
                             0.0+1.0j, 0.0+0.0j]], dtype=cp.complex64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
extraWorkspaceSizeInBytes = cusv.compute_expectation_batched_get_workspace_size(
    handle, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    d_matrices.data.ptr,
    cudaDataType.CUDA_C_32F, cusv.MatrixLayout.ROW, nMatrices,
    nBasisBits, ComputeType.COMPUTE_DEFAULT)

# allocate external workspace if necessary
if extraWorkspaceSizeInBytes > 0:
    workspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# compute expectation values
cusv.compute_expectation_batched(
    handle, d_svs.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride, exp_values.ctypes.data,
    d_matrices.data.ptr, cudaDataType.CUDA_C_32F, cusv.MatrixLayout.ROW, nMatrices,
    basisBits, nBasisBits, ComputeType.COMPUTE_32F, workspace_ptr, extraWorkspaceSizeInBytes)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(exp_values, expected):
    raise ValueError("results mismatch")
print("test passed")
