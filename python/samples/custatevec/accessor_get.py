# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)

bitOrdering = (2, 1)
maskBitString = (1,)
maskOrdering = (0,)
assert len(maskBitString) == len(maskOrdering)
maskLen = len(maskBitString)

bufferSize  = 3
accessBegin = 1
accessEnd   = 4

d_sv       = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
h_buf      = np.empty(bufferSize, dtype=np.complex64)
h_buf_res  = np.asarray([0.3+0.3j, 0.1+0.2j, 0.4+0.5j], dtype=np.complex64)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# create accessor and check the size of external workspace
accessor, workspace_size = cusv.accessor_create_view(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, bitOrdering, len(bitOrdering),
    maskBitString, maskOrdering, maskLen)

if workspace_size > 0:
    workspace = cp.cuda.alloc(workspace_size)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# set external workspace
cusv.accessor_set_extra_workspace(
    handle, accessor, workspace_ptr, workspace_size)

# get state vector components
cusv.accessor_get(
    handle, accessor, h_buf.ctypes.data, accessBegin, accessEnd)

# destroy accessor
cusv.accessor_destroy(accessor)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(h_buf, h_buf_res):
    raise ValueError("results mismatch")
else:
    print("test passed")
