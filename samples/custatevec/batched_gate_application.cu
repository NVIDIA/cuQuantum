/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecApplyMatrixBatched
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nSVs       = 2;
    const int nIndexBits = 3;
    const int nSvSize    = (1 << nIndexBits);
    const int svStride   = nSvSize;
    const int nTargets   = 1;
    const int nControls  = 2;
    const int adjoint    = 0;

    const int targets[]  = {2};
    const int controls[] = {0, 1};

    const int nMatrices = 2;
    const int matrixIndices[] = {1, 0};

    // 2 state vectors are allocated contiguously in single memory chunk.
    cuDoubleComplex h_svs[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5},
                                      { 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_svs_result[] = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, {-0.4,-0.5},
                                      { 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.4, 0.5}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.1, 0.2}};
    // 2 gate matrices are allocated contiguously in single memory chunk.
    cuDoubleComplex matrices[] = {{0.0, 0.0}, {1.0, 0.0},
                                  {1.0, 0.0}, {0.0, 0.0},
                                  {1.0, 0.0}, {0.0, 0.0},
                                  {0.0, 0.0}, {-1.0, 0.0}};

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
    HANDLE_ERROR( custatevecApplyMatrixBatchedGetWorkspaceSize(
                  handle, CUDA_C_64F, nIndexBits, nSVs, svStride,
                  CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrixIndices, matrices, CUDA_C_64F,
                  CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nMatrices, nTargets, nControls,
                  CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // apply gate
    HANDLE_ERROR( custatevecApplyMatrixBatched(
                  handle, d_svs, CUDA_C_64F, nIndexBits, nSVs, svStride,
                  CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrixIndices, matrices, CUDA_C_64F,
                  CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, nMatrices, targets, nTargets, controls,
                  nullptr, nControls, CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                  extraWorkspaceSizeInBytes) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //---------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_svs, d_svs, nSVs * svStride * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < nSVs * svStride; i++) {
        if (!almost_equal(h_svs[i], h_svs_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_svs) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("batched_gate_application example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("batched_gate_application example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
