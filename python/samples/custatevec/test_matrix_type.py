# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

import cupy as cp
import numpy as np

from cuquantum import custatevec as cusv
from cuquantum import cudaDataType, ComputeType


def run_test_matrix_type(
        handle, matrixType, matrix, matrixDataType, layout, nTargets,
        adjoint, computeType):
    # check the size of external workspace
    extraWorkspaceSizeInBytes = cusv.test_matrix_type_get_workspace_size(
        handle, matrixType, matrix.ctypes.data, matrixDataType, layout,
        nTargets, adjoint, computeType)

    # allocate external workspace if necessary
    if extraWorkspaceSizeInBytes > 0:
        extraWorkspace = cp.cuda.alloc(extraWorkspaceSizeInBytes)
        extraWorkspacePtr = extraWorkspace.ptr
    else:
        extraWorkspacePtr = 0

    # execute testing
    residualNorm = cusv.test_matrix_type(
        handle, matrixType, matrix.ctypes.data, matrixDataType, layout,
        nTargets, adjoint, computeType, extraWorkspacePtr, extraWorkspaceSizeInBytes)

    cp.cuda.Device().synchronize()

    return residualNorm


if __name__ == '__main__':
    nTargets = 1
    adjoint = 0

    # unitary and Hermitian matrix
    matrix = np.asarray([0.5+0.0j, 1/np.sqrt(2)-0.5j,
                         1/np.sqrt(2)+0.5j, -0.5+0.0j], dtype=np.complex128)

    # custatevec handle initialization
    handle = cusv.create()

    matrixDataType = cudaDataType.CUDA_C_64F
    layout = cusv.MatrixLayout.ROW
    computeType = ComputeType.COMPUTE_DEFAULT

    unitaryResidualNorm = run_test_matrix_type(handle, cusv.MatrixType.UNITARY, matrix,
                                               matrixDataType, layout, nTargets, adjoint,
                                               computeType)
    hermiteResidualNorm = run_test_matrix_type(handle, cusv.MatrixType.HERMITIAN, matrix,
                                               matrixDataType, layout, nTargets, adjoint,
                                               computeType)

    # destroy handle
    cusv.destroy(handle)

    correct = np.allclose(unitaryResidualNorm, 0.)
    correct &= np.allclose(hermiteResidualNorm, 0.)
    if correct:
        print("test_matrix_type example PASSED")
    else:
        raise RuntimeError("test_matrix_type example FAILED: wrong result")
