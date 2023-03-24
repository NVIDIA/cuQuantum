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
nControls  = 1

targets    = np.asarray([2], dtype=np.int32)
controls   = np.asarray([1], dtype=np.int32)
controlBitValues = np.asarray([1], dtype=np.int32)
paulis     = np.asarray([cusv.Pauli.Z], dtype=np.int32)

h_sv       = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
expected   = np.asarray([0.0+0.0j, 0.0+0.1j,-0.1+0.1j,-0.2+0.1j, 
                         0.2+0.2j, 0.3+0.3j, 0.4-0.3j, 0.5-0.4j], dtype=np.complex64)
d_sv = cp.asarray(h_sv)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# apply Pauli operator
cusv.apply_pauli_rotation(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, np.pi/2, paulis.ctypes.data,
    targets.ctypes.data, nTargets, controls.ctypes.data, controlBitValues.ctypes.data, nControls)

# destroy handle
cusv.destroy(handle)

# check result
if not cp.allclose(expected, d_sv):
    raise ValueError("results mismatch")
else:
    print("test passed")
