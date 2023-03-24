/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecTestMatrixType
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

double runTestMatrixType(custatevecHandle_t       handle,
                         custatevecMatrixType_t   matrixType,
                         const void*              matrix,
                         cudaDataType_t           matrixDataType,
                         custatevecMatrixLayout_t layout,
                         const uint32_t           nTargets,
                         const int32_t            adjoint,
                         custatevecComputeType_t  computeType) {

    double residualNorm;

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // check the size of external workspace
    HANDLE_ERROR( custatevecTestMatrixTypeGetWorkspaceSize(
                  handle, matrixType, matrix, matrixDataType, layout,
                  nTargets, adjoint, computeType, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // execute testing
    HANDLE_ERROR( custatevecTestMatrixType(
                  handle, &residualNorm, matrixType, matrix, matrixDataType, layout,
                  nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes) );

    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    return residualNorm;
}

int main(void) {

    const int nTargets = 1;
    const int adjoint = 0;

    // unitary and Hermitian matrix
    const double Rsqrt2 = 1. / std::sqrt(2.);
    cuDoubleComplex matrix[] = {{0.5, 0.0}, {Rsqrt2, -0.5},
                                {Rsqrt2, 0.5}, {-0.5, 0.0}};

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    cudaDataType_t matrixDataType = CUDA_C_64F;
    custatevecMatrixLayout_t layout = CUSTATEVEC_MATRIX_LAYOUT_ROW;
    custatevecComputeType_t computeType = CUSTATEVEC_COMPUTE_DEFAULT;

    double unitaryResidualNorm = runTestMatrixType(handle, CUSTATEVEC_MATRIX_TYPE_UNITARY, matrix,
                                                   matrixDataType, layout, nTargets, adjoint,
                                                   computeType) ;

    double hermiteResidualNorm = runTestMatrixType(handle, CUSTATEVEC_MATRIX_TYPE_HERMITIAN, matrix,
                                                   matrixDataType, layout, nTargets, adjoint,
                                                   computeType) ;

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    bool correct = true;

    correct &= almost_equal(unitaryResidualNorm, 0.);
    correct &= almost_equal(hermiteResidualNorm, 0.);

    if (correct) {
        printf("test_matrix_type example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("test_matrix_type example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}