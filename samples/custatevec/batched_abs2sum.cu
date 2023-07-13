/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecAbs2SumArrayBatched
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nSVs           = 2;
    const int nIndexBits     = 3;
    const int nSvElms        = (1 << nIndexBits);
    const int bitOrderingLen = 1;

    // square absolute values of state vector elements for 0/2-th bits will be summed up
    const int bitOrdering[] = {1};

    const custatevecIndex_t svStride = nSvElms;

    // 2 state vectors are allocated contiguously in single memory chunk.
    cuDoubleComplex h_svs[] = {{ 0.0,  0.0},  { 0.0,  0.1},  { 0.1,  0.1},  { 0.1,  0.2}, 
                               { 0.2,  0.2},  { 0.3,  0.3},  { 0.3,  0.4},  { 0.4,  0.5},
                               { 0.25, 0.25}, { 0.25, 0.25}, { 0.25, 0.25}, { 0.25, 0.25}, 
                               { 0.25, 0.25}, { 0.25, 0.25}, { 0.25, 0.25}, { 0.25, 0.25}};

    const custatevecIndex_t abs2sumStride = 2;
    const custatevecIndex_t batchedAbs2sumSize = nSVs * abs2sumStride;

    // abs2sum arrays are allocated contiguously in single memory chunk.
    double abs2sum[batchedAbs2sumSize];
    const double abs2sum_result[] = {0.27, 0.73, 0.5, 0.5};

    cuDoubleComplex *d_svs;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_svs, nSVs * nSvElms * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_svs, h_svs, nSVs * nSvElms * sizeof(cuDoubleComplex), 
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // compute abs2sum arrays
    HANDLE_ERROR( custatevecAbs2SumArrayBatched(
                  handle, d_svs, CUDA_C_64F, nIndexBits, nSVs, svStride, abs2sum, abs2sumStride,
                  bitOrdering, bitOrderingLen, nullptr, nullptr, 0) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaDeviceSynchronize() );

    bool correct = true;
    for (custatevecIndex_t i = 0; i < batchedAbs2sumSize; i++) {
        if (!almost_equal(abs2sum[i], abs2sum_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_svs) );

    if (correct) {
        printf("abs2sum_batched example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("abs2sum_batched example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}
