# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import cupy as cp

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType


nSVs         = 2
nIndexBits   = 3
svStride     = (1 << nIndexBits)

# 2 state vectors are allocated contiguously in single memory chunk.
d_svs        = cp.asarray([[0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j],
                           [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j, 
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j]], dtype=cp.complex64)

d_svs_res    = cp.asarray([[0.0+0.0j, 0.0+1.0j, 0.0+0.0j, 0.0+0.0j, 
                            0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j],
                           [0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
                            0.0+0.0j, 0.0+0.0j, 0.6+0.8j, 0.0+0.0j]], dtype=cp.complex64)

# 2 bitStrings are allocated contiguously in single memory chunk.
# The 1st SV collapses to |001> and the 2nd to |110>
# Note: bitStrings can also live on the host.
bitStrings   = cp.asarray([0b001, 0b110], dtype=cp.int64)

# bit ordering should only live on host.
bitOrdering  = np.asarray([0, 1, 2], dtype=np.int32)
bitStringLen = bitOrdering.size

# 2 norms are allocated contiguously in single memory chunk.
# Note: norms can also live on the host.
norms        = cp.asarray([0.01, 0.25], dtype=cp.float64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
extraWorkspaceSizeInBytes = cusv.collapse_by_bitstring_batched_get_workspace_size(
    handle, nSVs, bitStrings.data.ptr, norms.data.ptr)

# allocate external workspace if necessary
if extraWorkspaceSizeInBytes > 0:
    workspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# collapse the quantum states to the target bitstrings
cusv.collapse_by_bitstring_batched(
    handle, d_svs.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits, nSVs, svStride,
    bitStrings.data.ptr, bitOrdering.ctypes.data, bitStringLen, norms.data.ptr,
    workspace_ptr, extraWorkspaceSizeInBytes)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(d_svs_res, d_svs):
    raise ValueError("results mismatch")
print("test passed")
