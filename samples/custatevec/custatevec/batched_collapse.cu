/*
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuComplex.h>        // cuDoubleComplex
#include <custatevec.h>       // custatevecCollapseByBitStringBatched
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include "helper.hpp"         // HANDLE_ERROR, HANDLE_CUDA_ERROR

int main(void) {

    const int nSVs       = 2;
    const int nIndexBits = 3;
    const int svSize     = (1 << nIndexBits);
    const int svStride   = (1 << nIndexBits);  // no padding

    // 2 state vectors are allocated contiguously in single memory chunk.
    cuDoubleComplex h_svs[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5},
                                      { 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2}, 
                                      { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex h_svs_result[] = {{ 0.0, 0.0}, { 0.0, 1.0}, { 0.0, 0.0}, { 0.0, 0.0},
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0},
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0},
                                      { 0.0, 0.0}, { 0.0, 0.0}, { 0.6, 0.8}, { 0.0, 0.0}};

    // 2 bitStrings are allocated contiguously in single memory chunk.
    // The 1st SV collapses to |001> and the 2nd to |110>
    // Note: bitStrings can also live on the device.
    custatevecIndex_t bitStrings[] = {0b001, 0b110};

    // bit ordering should only live on host
    const int32_t bitOrdering[] = {0, 1, 2};
    const uint32_t bitStringLen = nIndexBits;

    // 2 norms are allocated contiguously in single memory chunk.
    // Note: norms can also live on the device.
    double norms[] = {0.01, 0.25};

    cuDoubleComplex *d_svs;
    HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_svs, nSVs * svSize * sizeof(cuDoubleComplex)) );
 
    HANDLE_CUDA_ERROR( cudaMemcpy(d_svs, h_svs, nSVs * svSize * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice) );

    //---------------------------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;
    HANDLE_ERROR( custatevecCreate(&handle) );

    void* extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // check the size of external workspace
    HANDLE_ERROR( custatevecCollapseByBitStringBatchedGetWorkspaceSize(
                  handle, nSVs, bitStrings, norms, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // collapse the quantum states to the target bitstrings
    HANDLE_ERROR( custatevecCollapseByBitStringBatched(
                  handle, d_svs, CUDA_C_64F, nIndexBits, nSVs, svStride,
                  bitStrings, bitOrdering, bitStringLen, norms,
                  extraWorkspace, extraWorkspaceSizeInBytes) );

    // destroy handle
    HANDLE_ERROR( custatevecDestroy(handle) );

    //---------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_svs, d_svs, nSVs * svSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < nSVs * svSize; i++) {
        if (!almost_equal(h_svs[i], h_svs_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_svs) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("batched_collapse example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("batched_collapse example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
