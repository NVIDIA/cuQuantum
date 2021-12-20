import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv


nIndexBits = 3
nSvSize    = (1 << nIndexBits)
nBasisBits = 1
maskLen    = 0
adjoint    = 0

basisBits  = [2]

d_sv       = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)
d_sv_res   = cp.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                         0.2-0.2j, 0.3-0.3j, 0.4-0.3j, 0.5-0.4j], dtype=np.complex64)
diagonals  = np.asarray([1.0+0.0j, 0.0-1.0j], dtype=np.complex64)

####################################################################################

# cuStateVec handle initialization
handle = cusv.create()

# check the size of external workspace
workspaceSize = cusv.apply_generalized_permutation_matrix_buffer_size(
    handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, 0, diagonals.ctypes.data, cuquantum.cudaDataType.CUDA_C_32F,
    basisBits, nBasisBits, maskLen)
if workspaceSize > 0:
    workspace = cp.cuda.memory.alloc(workspaceSize)
    workspace_ptr = workspace.ptr
else:
    workspace_ptr = 0

# apply matrix
cusv.apply_generalized_permutation_matrix(
    handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    0, diagonals.ctypes.data, cuquantum.cudaDataType.CUDA_C_32F, adjoint,
    basisBits, nBasisBits, 0, 0, maskLen,
    workspace_ptr, workspaceSize)

# destroy handle
cusv.destroy(handle)

# check result
if not np.allclose(d_sv, d_sv_res):
    raise ValueError("results mismatch")
else:
    print("test passed")
