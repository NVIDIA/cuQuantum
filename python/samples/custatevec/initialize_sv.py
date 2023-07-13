# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType


nIndexBits   = 3
svSize       = (1 << nIndexBits)

# populate the device memory with junk values (for illustrative purpose only)
# (we create a real random array of twice length, and view it as a complex array)
d_sv         = cp.random.random(2*svSize, dtype=cp.float32).view(cp.complex64)

d_sv_res     = cp.asarray([[1.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 
                            0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j]], dtype=cp.complex64)

###################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# initialize the state vector
cusv.initialize_state_vector(
    handle, d_sv.data.ptr, cudaDataType.CUDA_C_32F, nIndexBits,
    cusv.StateVectorType.ZERO) 

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(d_sv_res, d_sv):
    raise ValueError("results mismatch")
print("test passed")
