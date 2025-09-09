/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecInitializeStateVector
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nIndexBits = 3;
    const int svSize = (1 << nIndexBits);

    cuDoubleComplex h_sv[svSize];

    cuDoubleComplex h_sv_result[] = {{ 1.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0},
                                     { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, svSize * sizeof(cuDoubleComplex)) );

    // populate the device memory with junk values (for illustrative purpose only)
    HANDLE_CUDA_ERROR( cudaMemset(d_sv, 0x7F, svSize * sizeof(cuDoubleComplex)) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // initialize the state vector
    HANDLE_ERROR( custatevecInitializeStateVector(
                  handle, d_sv, CUDA_C_64F, nIndexBits, CUSTATEVEC_STATE_VECTOR_TYPE_ZERO) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv, d_sv, svSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < svSize; i++) {
        if (!almost_equal(h_sv[i], h_sv_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );

    if (correct) {
        printf("initialize_sv example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("initialize_sv example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
