/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecSwapIndexBits
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nIndexBits = 3;
    const int nSvSize    = (1 << nIndexBits);

    // swap 0th and 2nd qubits
    const int nBitSwaps  = 1;
    const int2 bitSwaps[] = {{0, 2}};

    // swap the state vector elements only if 1st qubit is 1
    const int maskLen = 1;
    int maskBitString[] = {1};
    int maskOrdering[] = {1};

    // 0.2|001> + 0.4|011> - 0.4|101> - 0.8|111>
    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.2, 0.0}, { 0.0, 0.0}, { 0.4, 0.0}, 
                                     { 0.0, 0.0}, {-0.4, 0.0}, { 0.0, 0.0}, {-0.8, 0.0}};

    // 0.2|001> + 0.4|110> - 0.4|101> - 0.8|111>
    cuDoubleComplex h_sv_result[] = {{ 0.0, 0.0}, { 0.2, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, 
                                     { 0.0, 0.0}, {-0.4, 0.0}, { 0.4, 0.0}, {-0.8, 0.0}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc(&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), 
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // bit swap
    HANDLE_ERROR( custatevecSwapIndexBits(
                  handle, d_sv, CUDA_C_64F, nIndexBits, bitSwaps, nBitSwaps,
                  maskBitString, maskOrdering, maskLen) );

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

    if (correct) {
        printf("swap_index_bits example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("swap_index_bits example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
