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

    const int nPauliOperatorArrays = 2;
    const custatevecPauli_t pauliOperatorsI[] = {CUSTATEVEC_PAULI_I};
    const custatevecPauli_t pauliOperatorsXY[] = {CUSTATEVEC_PAULI_X, CUSTATEVEC_PAULI_Y};
    const custatevecPauli_t* pauliOperatorsArray[] = {pauliOperatorsI, pauliOperatorsXY};

    const unsigned nBasisBitsArray[] = {1, 2};
    const int basisBitsI[] = {1};
    const int basisBitsXY[] = {1, 2};
    const int* basisBitsArray[] = {basisBitsI, basisBitsXY};

    double expectationValues[nPauliOperatorArrays];
    double expectationValues_result[] = {1.0, -0.14};
    cuDoubleComplex h_sv[]  = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2},
                               { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};

    cuDoubleComplex *d_sv;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_sv, nSvSize * sizeof(cuDoubleComplex)) );

    HANDLE_CUDA_ERROR( cudaMemcpy(d_sv, h_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice) );

    //----------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    // apply Pauli operator
    HANDLE_ERROR( custatevecComputeExpectationsOnPauliBasis(
                  handle, d_sv, CUDA_C_64F, nIndexBits, expectationValues,
                  pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    bool correct = true;
    for (int i = 0; i < nPauliOperatorArrays; i++) {
        if (!almost_equal(expectationValues[i], expectationValues_result[i]) ) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );

    if (correct) {
        printf("example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
