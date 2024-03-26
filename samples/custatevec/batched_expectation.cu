/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecComputeExpectationBatched
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nSVs       = 2;
    const int nIndexBits = 3;
    const int nSvSize    = (1 << nIndexBits);
    const int svStride   = nSvSize;
    const int nBasisBits = 1;

    const int basisBits[]  = {2};
    const int nMatrices = 2;

    cuDoubleComplex expectationValue[4];
    cuDoubleComplex expectationValueResult[] = {{0.48, 0.0}, {-0.10, 0.0}, {0.46, 0.0}, {-0.16, 0.0}};

    // 2 state vectors are allocated contiguously in single memory chunk.
    cuDoubleComplex h_svs[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5},
                                      { 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.5, 0.4}};
    // 2 gate matrices are allocated contiguously in single memory chunk.
    cuDoubleComplex matrices[] = {{0.0, 0.0}, {1.0, 0.0},
                                  {1.0, 0.0}, {0.0, 0.0},
                                  {0.0, 0.0}, {0.0, -1.0},
                                  {0.0, 1.0}, {0.0, 0.0}};

    cuDoubleComplex *d_svs;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_svs, nSVs * svStride * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_svs, h_svs, nSVs * svStride * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice) );

    //---------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // check the size of external workspace
    HANDLE_ERROR( custatevecComputeExpectationBatchedGetWorkspaceSize(
                  handle, CUDA_C_64F, nIndexBits, nSVs, svStride,
                  matrices, CUDA_C_64F,
                  CUSTATEVEC_MATRIX_LAYOUT_ROW, nMatrices, nBasisBits,
                  CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // compute expectation values
    HANDLE_ERROR( custatevecComputeExpectationBatched(
                  handle, d_svs, CUDA_C_64F, nIndexBits, nSVs, svStride,
                  expectationValue, matrices, CUDA_C_64F,
                  CUSTATEVEC_MATRIX_LAYOUT_ROW, nMatrices, basisBits, nBasisBits,
                  CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                  extraWorkspaceSizeInBytes) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //---------------------------------------------------------------------------------------------

    bool correct = true;
    for (int i = 0; i < nSVs * nMatrices; i++) {
        if (!almost_equal(expectationValue[i], expectationValueResult[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_svs) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("batched_expectation example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("batched_expectation example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}