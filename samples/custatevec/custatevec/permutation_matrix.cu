/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrix
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nIndexBits = 3;
    const int nSvSize    = (1 << nIndexBits);
    const int nBasisBits = 2;
    const int maskLen    = 1;
    const int adjoint    = 0;

    const int basisBits[]     = {0, 1};
    const int maskOrdering[]  = {2};
    const int maskBitString[] = {1};
    custatevecIndex_t permutation[] = {0, 2, 1, 3};

    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2},
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_sv_result[] = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2},
                                     { 0.2, 0.2}, {-0.4, 0.3}, {-0.3, 0.3}, { 0.4, 0.5}};
    cuDoubleComplex diagonals[] = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // check the size of external workspace
    HANDLE_ERROR( custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
                  handle, CUDA_C_64F, nIndexBits, permutation, diagonals, CUDA_C_64F, basisBits,
                  nBasisBits, maskLen, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // apply matrix
    HANDLE_ERROR( custatevecApplyGeneralizedPermutationMatrix(
                  handle, d_sv, CUDA_C_64F, nIndexBits, permutation, diagonals, CUDA_C_64F,
                  adjoint, basisBits, nBasisBits, maskOrdering, maskBitString, maskLen,
                  extraWorkspace, extraWorkspaceSizeInBytes) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < nSvSize; i++) {
        if (!almost_equal(h_sv[i], h_sv_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("permutation_matrix example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("permutation_matrix example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
