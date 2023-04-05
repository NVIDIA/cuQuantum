# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)
nBasisBits = 1

basisBits  = np.asarray([1], dtype=np.int32)

h_sv       = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
d_sv       = cp.asarray(h_sv)

# the gate matrix can live on either host (np) or device (cp)
matrix     = cp.asarray([1.0+0.0j, 2.0+1.0j, 2.0-1.0j, 3.0+0.0j], dtype=np.complex64)
if isinstance(matrix, cp.ndarray):
    matrix_ptr = matrix.data.ptr
elif isinstance(matrix, np.ndarray):
    matrix_ptr = matrix.ctypes.data
else:
    raise ValueError

# expectation values must stay on host
expect     = np.empty((2,), dtype=np.float64)
expect_expected = np.asarray([4.1, 0.0], dtype=np.float64)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
workspaceSize = cusv.compute_expectation_get_workspace_size(
    handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
    cusv.MatrixLayout.ROW, nBasisBits, cuquantum.ComputeType.COMPUTE_32F)
if workspaceSize > 0:
    workspace = cp.cuda.memory.alloc(workspaceSize)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply gate
cusv.compute_expectation(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    expect.ctypes.data, cuquantum.cudaDataType.CUDA_C_64F,
    matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F, cusv.MatrixLayout.ROW,
    basisBits.ctypes.data, nBasisBits,
    cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

# destroy handle
cusv.destroy(handle)

# check result
if not np.allclose(expect, expect_expected, atol=1E-6):
    raise ValueError("results mismatch")
else:
    print("test passed")
