# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize = (1 << nIndexBits)
adjoint = 0

targets  = [2]
n_targets = 1
n_controls = 0

d_sv       = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
d_sv_res   = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2-0.2j, 0.3-0.3j, 0.4-0.3j, 0.5-0.4j], dtype=np.complex64)
diagonals  = np.asarray([1.0+0.0j, 0.0-1.0j], dtype=np.complex64)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
workspaceSize = cusv.apply_generalized_permutation_matrix_get_workspace_size(
    handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, 0, diagonals.ctypes.data, cuquantum.cudaDataType.CUDA_C_32F,
    targets, n_targets, n_controls)
if workspaceSize > 0:
    workspace = cp.cuda.memory.alloc(workspaceSize)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply matrix
cusv.apply_generalized_permutation_matrix(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    0, diagonals.ctypes.data, cuquantum.cudaDataType.CUDA_C_32F, adjoint,
    targets, n_targets, 0, 0, n_controls,
    workspace_ptr, workspaceSize)

# destroy handle
cusv.destroy(handle)

# check result
if not np.allclose(d_sv, d_sv_res):
    raise ValueError("results mismatch")
else:
    print("test passed")
