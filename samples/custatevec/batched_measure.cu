/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecMeasureBatched
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nSVs         = 2;
    const int nIndexBits   = 3;
    const int nSvElms      = (1 << nIndexBits);
    const int bitStringLen = 3;

    const int bitOrdering[] = {2, 1, 0};

    const custatevecIndex_t svStride = nSvElms;

    custatevecIndex_t bitStrings[nSVs];
    const custatevecIndex_t bitStrings_result[] = {0b100, 0b011};

    // In real appliction, random number in range [0, 1) will be used.
    const double randnums[] = {0.009, 0.5}; 

    // 2 state vectors are allocated contiguously in single memory chunk.
    cuDoubleComplex h_svs[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5},
                                      { 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_svs_result[] = {{ 0.0, 0.0}, { 0.0, 1.0}, { 0.0, 0.0}, { 0.0, 0.0}, 
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0},
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, 
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.6, 0.8}, { 0.0, 0.0}};

    cuDoubleComplex *d_svs;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_svs, nSVs * nSvElms * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_svs, h_svs, nSVs * nSvElms * sizeof(cuDoubleComplex), 
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // batched measurement
    HANDLE_ERROR( custatevecMeasureBatched(
                  handle, d_svs, CUDA_C_64F, nIndexBits, nSVs, svStride, bitStrings, bitOrdering,
                  bitStringLen, randnums, CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_svs, d_svs, nSVs * nSvElms * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < nSVs * nSvElms; i++) {
        if (!almost_equal(h_svs[i], h_svs_result[i])) {
            correct = false;
            break;
        }
    }

    for (int i = 0; i < nSVs; i++) {
        if (bitStrings[i] != bitStrings_result[i]) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_svs) );

    if (correct) {
        printf("measure_batched example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("measure_batched example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }
}
