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

    const int nIndexBits   = 3;
    const int nSvSize      = (1 << nIndexBits);
    const int bitStringLen = 3;

    const int bitOrdering[] = {2, 1, 0};

    int bitString[bitStringLen];
    const int bitString_result[] = {1, 1, 0};

    // In real appliction, random number in range [0, 1) will be used.
    const double randnum = 0.5; 

    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_sv_result[] = {{ 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, 
                                     { 0.0, 0.0}, { 0.0, 0.0}, { 0.6, 0.8}, { 0.0, 0.0}};
   
    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), 
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // batch measurement
    HANDLE_ERROR( custatevecBatchMeasure(
                  handle, d_sv, CUDA_C_64F, nIndexBits, bitString, bitOrdering, bitStringLen, 
                  randnum, CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO) );

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

    for (int i = 0; i < bitStringLen; i++) {
        if (bitString[i] != bitString_result[i]) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );

    if (correct) {
        printf("batch_measure example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("batch_measure example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}

