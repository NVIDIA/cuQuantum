# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)

bitOrdering = (1, 2, 0)
maskLen = 0

d_sv       = cp.zeros(nSvSize, dtype=np.complex64)
d_sv_res   = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
h_buf      = np.asarray([0.0+0.0j, 0.1+0.1j, 0.2+0.2j, 0.3+0.4j,
                         0.0+0.1j, 0.1+0.2j, 0.3+0.3j, 0.4+0.5j], dtype=np.complex64)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# create accessor and check the size of external workspace
accessor, workspace_size = cusv.accessor_create(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, bitOrdering, len(bitOrdering),
    0, 0, maskLen)

if workspace_size > 0:
    workspace = cp.cuda.alloc(workspace_size)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# set external workspace
cusv.accessor_set_extra_workspace(
    handle, accessor, workspace_ptr, workspace_size)

# set state vector components
cusv.accessor_set(
    handle, accessor, h_buf.ctypes.data, 0, nSvSize)

# destroy accessor
cusv.accessor_destroy(accessor)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(d_sv, d_sv_res):
    raise ValueError("results mismatch")
else:
    print("test passed")
