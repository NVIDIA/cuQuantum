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
    const int nBasisBits = 1;

    const int basisBits[] = {1};

    cuDoubleComplex expectationValue;
    cuDoubleComplex expectationValueResult = {4.1, 0.0};

    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex matrix[] = {{1.0, 0.0}, {2.0, 1.0},
                                {2.0,-1.0}, {3.0, 0.0}};

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
    HANDLE_ERROR( custatevecComputeExpectationGetWorkspaceSize(
                  handle, CUDA_C_64F, nIndexBits, matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                  nBasisBits, CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // compute expectation
    HANDLE_ERROR( custatevecComputeExpectation(
                  handle, d_sv, CUDA_C_64F, nIndexBits, &expectationValue, CUDA_C_64F, nullptr,
                  matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, basisBits, nBasisBits,
                  CUSTATEVEC_COMPUTE_64F, extraWorkspace, extraWorkspaceSizeInBytes) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    if ( !almost_equal(expectationValue, expectationValueResult) ) {
        correct = false;
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("expectation example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("expectation example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
