# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
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
adjoint       = 0

targets       = [2]
nTargets      = len(targets)
controls      = [0, 1]
nControls     = len(controls)

matrixIndices = [1, 0]
nMatrices     = len(matrixIndices)

# 2 state vectors are allocated contiguously in single memory chunk.
d_svs         = cp.asarray([[0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                             0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j],
                            [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                             0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j]], dtype=cp.complex64)

d_svs_res     = cp.asarray([[0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, -0.4-0.5j],
                           [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.4+0.5j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.1+0.2j]], dtype=cp.complex64)

# 2 gate matrices are allocated contiguously in single memory chunk.
# Note: gate matrices can also live on the host.
d_matrices    = cp.asarray([[0.0+0.0j, 1.0+0.0j, 
                             1.0+0.0j, 0.0+0.0j],
                            [1.0+0.0j, 0.0+0.0j, 
                             0.0+0.0j, -1.0+0.0j]], dtype=cp.complex64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
extraWorkspaceSizeInBytes = cusv.apply_matrix_batched_get_workspace_size(
    handle, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    cusv.MatrixMapType.MATRIX_INDEXED, matrixIndices, d_matrices.data.ptr,
    cudaDataType.CUDA_C_32F, cusv.MatrixLayout.ROW, adjoint, nMatrices,
    nTargets, nControls,
    ComputeType.COMPUTE_32F)

# allocate external workspace if necessary
if extraWorkspaceSizeInBytes > 0:
    workspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply gate
cusv.apply_matrix_batched(
    handle, d_svs.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    cusv.MatrixMapType.MATRIX_INDEXED, matrixIndices, d_matrices.data.ptr,
    cudaDataType.CUDA_C_32F, cusv.MatrixLayout.ROW, adjoint, nMatrices,
    targets, nTargets, controls, 0, nControls,
    ComputeType.COMPUTE_32F, workspace_ptr, extraWorkspaceSizeInBytes)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(d_svs_res, d_svs):
    raise ValueError("results mismatch")
print("test passed")
