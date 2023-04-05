# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)
nTargets   = 1
nControls  = 2
adjoint    = 0

targets    = np.asarray([2], dtype=np.int32)
controls   = np.asarray([0, 1], dtype=np.int32)

h_sv       = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
expected   = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.4+0.5j, 
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.1+0.2j], dtype=np.complex64)

# the gate matrix can live on either host (np) or device (cp)
matrix     = cp.asarray([0.0+0.0j, 1.0+0.0j, 1.0+0.0j, 0.0+0.0j], dtype=np.complex64)
if isinstance(matrix, cp.ndarray):
    matrix_ptr = matrix.data.ptr
elif isinstance(matrix, np.ndarray):
    matrix_ptr = matrix.ctypes.data
else:
    raise ValueError

d_sv = cp.asarray(h_sv)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()
workspaceSize = cusv.apply_matrix_get_workspace_size(
    handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
    cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

# check the size of external workspace
if workspaceSize > 0:
    workspace = cp.cuda.memory.alloc(workspaceSize)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply gate
cusv.apply_matrix(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
    cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
    cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(expected, d_sv):
    raise ValueError("results mismatch")
else:
    print("test passed")
