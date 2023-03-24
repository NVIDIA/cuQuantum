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

    const int bitOrderingLen = 2;
    const int bitOrdering[]  = {2, 1};

    const int maskLen         = 1;
    const int maskBitString[] = {1};
    const int maskOrdering[]  = {0};

    const int bufferSize  = 3;
    const int accessBegin = 1;
    const int accessEnd   = 4;

    cuDoubleComplex h_sv[]        = {{ 0.0, 0.0}, { 0.0, 0.1}, { 0.1, 0.1}, { 0.1, 0.2},
                                     { 0.2, 0.2}, { 0.3, 0.3}, { 0.3, 0.4}, { 0.4, 0.5}};
    cuDoubleComplex buffer[]        = {{ 0.0, 0.0}, { 0.0, 0.0}, { 0.0, 0.0}};
    cuDoubleComplex buffer_result[] = {{ 0.3, 0.3}, { 0.1, 0.2}, { 0.4, 0.5}};

    custatevecAccessorDescriptor_t accessor;

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

    // create accessor and check the size of external workspace
    HANDLE_ERROR( custatevecAccessorCreateView(
                  handle, d_sv, CUDA_C_64F, nIndexBits, &accessor, bitOrdering, bitOrderingLen,
                  maskBitString, maskOrdering, maskLen, &extraWorkspaceSizeInBytes) );

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        HANDLE_CUDA_ERROR( cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes) );

    // set external workspace
    HANDLE_ERROR( custatevecAccessorSetExtraWorkspace(
                  handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes) );

    // get state vector components
    HANDLE_ERROR( custatevecAccessorGet(
                  handle, accessor, buffer, accessBegin, accessEnd) );

    // destroy descriptor and handle
    HANDLE_ERROR( custatevecAccessorDestroy(accessor) );
    HANDLE_ERROR( custatevecDestroy(handle) );

    //----------------------------------------------------------------------------------------------

    HANDLE_CUDA_ERROR( cudaMemcpy(h_sv, d_sv, nSvSize * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) );

    bool correct = true;
    for (int i = 0; i < bufferSize; i++) {
        if (!almost_equal(buffer[i], buffer_result[i])) {
            correct = false;
            break;
        }
    }

    HANDLE_CUDA_ERROR( cudaFree(d_sv) );
    if (extraWorkspaceSizeInBytes)
        HANDLE_CUDA_ERROR( cudaFree(extraWorkspace) );

    if (correct) {
        printf("accessor_get example PASSED\n");
        return EXIT_SUCCESS;
    }
    else {
        printf("accessor_get example FAILED: wrong result\n");
        return EXIT_FAILURE;
    }

}
