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
#include <cmath>              // acos

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nIndexBits = 3;
    const int nSvSize    = (1 << nIndexBits);
    const int nTargets   = 1;
    const int nControls  = 1;

    const int targets[]  = {2};
    const int controls[] = {1};
    const int controlBitValues[] = {1};

    const double pi = std::acos(-1.0);

    const custatevecPauli_t paulis[] = {CUSTATEVEC_PAULI_Z};

    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_sv_result[] = {{ 0.0, 0.0}, { 0.0, 0.1}, {-0.1, 0.1}, {-0.2, 0.1}, 
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.4,-0.3}, { 0.5,-0.4}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex), 
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // apply Pauli operator
    HANDLE_ERROR( custatevecApplyPauliRotation(
                  handle, d_sv, CUDA_C_64F, nIndexBits, pi / 2.0, paulis, targets, nTargets, 
                  controls, controlBitValues, nControls) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < nSvSize; i++) {
        if ( !almost_equal(h_sv[i], h_sv_result[i]) ) {
            correct = false;
             break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );

    if (correct) {
        printf("exponential_pauli example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("exponential_pauli example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
