# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


nIndexBits = 3
nSvSize = (1 << nIndexBits)

# swap 0th and 2nd qubits
nBitSwaps = 1
bitSwaps = [(0, 2)]

# swap the state vector elements only if 1st qubit is 1
maskLen = 1;
maskBitString = [1]
maskOrdering = [1]

# 0.2|001> + 0.4|011> - 0.4|101> - 0.8|111>
sv = cp.asarray([0.0+0.0j,  0.2+0.0j, 0.0+0.0j,  0.4+0.0j, 
                 0.0+0.0j, -0.4+0.0j, 0.0+0.0j, -0.8+0.0j],
                dtype=cp.complex128)

# 0.2|001> + 0.4|110> - 0.4|101> - 0.8|111>
sv_result = cp.asarray([0.0+0.0j,  0.2+0.0j, 0.0+0.0j,  0.0+0.0j, 
                        0.0+0.0j, -0.4+0.0j, 0.4+0.0j, -0.8+0.0j],
                       dtype=cp.complex128)

# custatevec handle initialization
handle = cusv.create()

# bit swap
cusv.swap_index_bits(
    handle, sv.data.ptr, cudaDataType.CUDA_C_64F, nIndexBits,
    bitSwaps, nBitSwaps,
    maskBitString, maskOrdering, maskLen)

# destroy handle
cusv.destroy(handle)

correct = cp.allclose(sv, sv_result)
if correct:
    print("swap_index_bits example PASSED")
else:
    raise RuntimeError("swap_index_bits example FAILED: wrong result")
